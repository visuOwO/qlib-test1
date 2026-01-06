import qlib
import argparse
import pandas as pd
import numpy as np
import traceback
import re
from pathlib import Path
from typing import Optional, Union
from qlib.data import D
from qlib.config import C

_WORKER_CACHE = {}
DEFAULT_CSI500_MEMBERSHIP = Path("./qlib_meta/csi500_membership.csv")
_VWAP_EXPR = "Div($amount, $volume)"


def expand_vwap_expression(expression: str) -> str:
    if "$vwap" not in expression:
        return expression
    return re.sub(r"\$vwap\b", _VWAP_EXPR, expression)


def expand_vwap_expressions(expressions):
    expanded = []
    rename_map = {}
    for expr in expressions:
        expanded_expr = expand_vwap_expression(expr)
        expanded.append(expanded_expr)
        if expanded_expr != expr:
            rename_map[expanded_expr] = expr
    return expanded, rename_map

def _ts_code_to_qlib_instrument(ts_code: str) -> str:
    parts = ts_code.split(".")
    if len(parts) != 2:
        return ts_code.upper()
    return f"{parts[1]}{parts[0]}".upper()

def _load_csi500_membership(membership_path: Optional[Union[str, Path]]):
    if not membership_path:
        return None
    path = Path(membership_path)
    if not path.exists():
        print(f"[WARN] CSI500 membership file not found: {path}")
        return None

    df = pd.read_csv(path)
    if df.empty:
        print(f"[WARN] CSI500 membership file is empty: {path}")
        return None

    if "instrument" not in df.columns:
        if "con_code" in df.columns:
            df["instrument"] = df["con_code"].apply(_ts_code_to_qlib_instrument)
        else:
            print(f"[WARN] CSI500 membership missing instrument/con_code: {path}")
            return None

    if "date" in df.columns:
        df["datetime"] = pd.to_datetime(df["date"])
    elif "trade_date" in df.columns:
        df["datetime"] = pd.to_datetime(df["trade_date"])
    else:
        print(f"[WARN] CSI500 membership missing date column: {path}")
        return None

    df["instrument"] = df["instrument"].astype(str).str.upper()
    return pd.MultiIndex.from_frame(df[["instrument", "datetime"]])

def _align_membership_index(index: pd.MultiIndex, membership_index: pd.MultiIndex) -> pd.Series:
    if membership_index is None or index.empty:
        return pd.Series(False, index=index)

    mem_df = membership_index.to_frame(index=False).rename(columns={"datetime": "mem_date"})
    mem_dates = mem_df[["mem_date"]].drop_duplicates().sort_values("mem_date")
    dates = pd.DataFrame(
        {"datetime": pd.to_datetime(index.get_level_values("datetime").unique())}
    ).sort_values("datetime")
    mapped = pd.merge_asof(
        dates,
        mem_dates,
        left_on="datetime",
        right_on="mem_date",
        direction="backward",
    )
    date_to_mem = mapped.set_index("datetime")["mem_date"]
    mem_date_for_row = index.get_level_values("datetime").map(date_to_mem)
    membership_set = set(zip(mem_df["instrument"], mem_df["mem_date"]))
    mask = [
        (inst, mem_date) in membership_set if pd.notna(mem_date) else False
        for inst, mem_date in zip(index.get_level_values("instrument"), mem_date_for_row)
    ]
    return pd.Series(mask, index=index)

def init_worker(provider_uri, start_date, end_date, benchmark_code, csi500_membership_path=None):
    """
    子进程初始化函数：只运行一次
    负责初始化 Qlib 并加载公共数据到全局变量
    """
    try:
        if not C.get("initialized", False):
            qlib.init(provider_uri=provider_uri, region=qlib.constant.REG_CN)
        
        # 1. 预加载基准收益
        bench_df = D.features([benchmark_code], ["Ref($close, -5) / Ref($close, -1) - 1"], start_date, end_date)
        if not bench_df.empty and isinstance(bench_df.index, pd.MultiIndex):
            bench_df = bench_df.droplevel(0)
            
        # 2. 预加载目标收益 (Target)
        target_df = D.features(D.instruments("all"), ["Ref($close, -5) / Ref($close, -1) - 1"], start_time=start_date, end_time=end_date, freq="day")
        target_df.columns = ['target']
        
        # 3. 预加载基础行情 (可选，用于计算相关性，防止每次都去读 $close)
        # 这里我们至少可以把 target 和 benchmark 存起来
        _WORKER_CACHE['benchmark_ret'] = bench_df
        _WORKER_CACHE['target_df'] = target_df

        csi500_index = _load_csi500_membership(csi500_membership_path or DEFAULT_CSI500_MEMBERSHIP)
        _WORKER_CACHE["csi500_index"] = csi500_index
        
        print(f"[Worker] Initialized with Target Data shape: {target_df.shape}")
        
    except Exception as e:
        print(f"[Worker Init Error] {e}")
        traceback.print_exc()

def evaluate_factor_mp(factor_expression: str, start_date: str, end_date: str) -> tuple:
    """
    优化后的评估函数：使用 _WORKER_CACHE
    注意：参数减少了，因为部分数据在 init_worker 里加载了
    """
    try:
        # 从全局缓存获取数据
        target_df = _WORKER_CACHE.get('target_df')
        benchmark_ret_df = _WORKER_CACHE.get('benchmark_ret')
        csi500_index = _WORKER_CACHE.get("csi500_index")
        
        if target_df is None or target_df.empty:
            return -1.0, -1.0, -1.0, -1.0, -1.0

        # 4. Get factor data (因子数据必须动态计算)
        try:
            # 注意：这里我们依然需要 $close 和 $industry
            # Qlib 的 D.features 会处理对齐，所以这里重新读取 $close 开销是可以接受的，或者也可以进一步优化缓存
            expanded_expr, rename_map = expand_vwap_expressions([factor_expression])
            factor_df = D.features(
                D.instruments("all"),
                expanded_expr + ["$close", "$industry", "$circ_mv"],
                start_time=start_date,
                end_time=end_date,
                freq="day"
            )
            if rename_map:
                factor_df = factor_df.rename(columns=rename_map)
        except Exception as e:
            # 公式错误等情况
            return -1.0, -1.0, -1.0, -1.0, -1.0

        if factor_df.empty: return -1.0, -1.0, -1.0, -1.0, -1.0
        
        if 'SH000300' in factor_df.index.get_level_values('instrument'):
            factor_df = factor_df.drop('SH000300', level='instrument')

        # 5. Merge (使用缓存的 target_df)
        merged_df = pd.merge(factor_df, target_df, left_index=True, right_index=True, how='inner')
        merged_df.dropna(inplace=True)
        
        if merged_df.empty: return -1.0, -1.0, -1.0, -1.0, -1.0

        # 6. Circulating Market Value Neutralization (use full market)
        merged_df['log_mkt_cap'] = np.log(merged_df['$circ_mv'] + 1)

        def clip_outliers(series):
            mean = series.mean()
            std = series.std()
            return series.clip(mean - 3*std, mean + 3*std)
        
        def regress_out_size(df_day):
            # Y: 因子值, X: 对数市值
            Y = df_day[factor_expression].values
            X = df_day['log_mkt_cap'].values
            
            # 处理 NaN
            valid_mask = ~np.isnan(Y) & ~np.isnan(X)
            if np.sum(valid_mask) < 10: # 如果有效数据太少，直接填0
                return pd.Series(0, index=df_day.index)
            
            Y_valid = Y[valid_mask]
            X_valid = X[valid_mask]
            
            # 简单的单变量回归 slope = cov(x,y) / var(x)
            x_mean = np.mean(X_valid)
            y_mean = np.mean(Y_valid)
            numerator = np.sum((X_valid - x_mean) * (Y_valid - y_mean))
            denominator = np.sum((X_valid - x_mean) ** 2)
            
            if denominator == 0:
                slope = 0
            else:
                slope = numerator / denominator
                
            intercept = y_mean - slope * x_mean
            
            # 计算残差
            resid = Y - (slope * X + intercept)
            
            # 将残差填回对应的索引
            return pd.Series(resid, index=df_day.index)
        
        # 去极值
        merged_df['raw_factor'] = merged_df.groupby(level='datetime')[factor_expression].transform(clip_outliers)
        merged_df['raw_factor'] = merged_df['raw_factor'].fillna(0)

        # 每日市值中性化 (Regress out Size)
        merged_df['size_neu_factor'] = merged_df.groupby(level='datetime', group_keys=False).apply(regress_out_size)

        def industry_neutralize_and_standardize(df_group):
            return (df_group - df_group.mean()) / (df_group.std() + 1e-9)

        # 8. Industry Neutralization
        merged_df['neu_factor'] = merged_df.groupby(['datetime', '$industry'])['size_neu_factor'].transform(industry_neutralize_and_standardize)
        merged_df['neu_factor'] = merged_df['neu_factor'].fillna(0)

        # 7. Filter to CSI500 for factor analysis
        analysis_df = merged_df
        if csi500_index is not None:
            in_membership = _align_membership_index(merged_df.index, csi500_index)
            analysis_df = merged_df.loc[in_membership.values]
        if analysis_df.empty:
            return -1.0, -1.0, -1.0, -1.0, -1.0

        # 8. Price Correlation Check
        price_corr_series = analysis_df.groupby(level='datetime').apply(
            lambda x: x[factor_expression].corr(x['$close'], method='spearman')
        )
        price_corr = price_corr_series.mean()
        if np.isnan(price_corr): price_corr = 1.0

        # 9. Calculate Metrics
        ic_series = analysis_df.groupby(level='datetime').apply(
            lambda x: x['neu_factor'].corr(x['target'], method='spearman')
        )
        ic_mean = ic_series.mean()
        ic_std = ic_series.std()

        if np.isnan(ic_mean):
            ic_mean = -1.0

        # 计算环节出问题（如标准差为0/NaN），ICIR 置 0；其他异常会统一返回 -1
        if ic_std == 0 or np.isnan(ic_std):
            icir = 0
        else:
            icir = ic_mean / ic_std

        def get_group_return(df_day):
            try:
                df_day['group'] = pd.qcut(df_day['neu_factor'], 5, labels=False, duplicates='drop')
                return df_day.groupby('group')['target'].mean()
            except ValueError:
                return pd.Series()
        
        daily_group_returns = analysis_df.groupby(level='datetime').apply(get_group_return)
        
        if isinstance(daily_group_returns, pd.Series):
            if isinstance(daily_group_returns.index, pd.MultiIndex):
                daily_group_returns = daily_group_returns.unstack()
            else:
                daily_group_returns = daily_group_returns.to_frame().T
        
        # 统一补全 5 档列，缺失填 NaN
        daily_group_returns = daily_group_returns.reindex(columns=range(5))
        if 4 not in daily_group_returns.columns:
            return -1.0, ic_mean, icir, price_corr, 0.0

        # === 计算日度 Spearman 单调性并取均值 ===
        def daily_monotone(row):
            # 如果当日某组缺失，则放弃该日
            if row.isna().any():
                return np.nan
            return row.corr(pd.Series(range(5)), method="spearman")

        mono_series = daily_group_returns.apply(daily_monotone, axis=1)
        monotonicity = mono_series.dropna().mean() if not mono_series.empty else 0.0
        if np.isnan(monotonicity): monotonicity = 0.0

        long_ret = daily_group_returns[4]
        
        if benchmark_ret_df.empty:
            bench_ret = pd.Series(0, index=long_ret.index)
        else:
            bench_ret = benchmark_ret_df.reindex(long_ret.index).fillna(0).iloc[:, 0]

        excess_ret = long_ret - bench_ret
        mean_excess = excess_ret.mean()
        std_excess = excess_ret.std()

        if std_excess == 0 or np.isnan(std_excess):
            ir = -1.0
        else:
            ir = (mean_excess / std_excess) * np.sqrt(252)
            if np.isnan(ir): ir = -1.0
        
        return ir, ic_mean, icir, price_corr, monotonicity

    except Exception:
        # 建议打印错误堆栈，否则调试困难
        traceback.print_exc() 
        return -1.0, -1.0, -1.0, -1.0, -1.0


class FactorMiningEnv:
    def __init__(self, start_date, end_date, benchmark_code="sh000300", provider_uri="./qlib_bin_data", csi500_membership_path=DEFAULT_CSI500_MEMBERSHIP):
        self.start_date = start_date
        self.end_date = end_date
        self.benchmark_code = benchmark_code
        self.provider_uri = provider_uri
        self.csi500_membership_path = csi500_membership_path
        
        # Initialize Qlib in the main process
        if not C.get("initialized", False):
            qlib.init(provider_uri=self.provider_uri, region=qlib.constant.REG_CN)
            print(f"Qlib initialized with data from: {provider_uri}")

        self.raw_features = [
            "$open", "$high", "$low", "$close", "$vwap", "$volume", "$amount", 
            "$turnover_rate", "$turnover_rate_f", "$volume_ratio",
            "$pe_ttm", "$pb", "$ps_ttm", "$dv_ttm",
            "$total_mv", "$circ_mv"
        ]
        # Benchmark return is no longer pre-calculated for the class instance
        # It will be calculated in each worker process

    def evaluate_factor(self, factor_expression: str) -> tuple:
        """
        This method now acts as a wrapper for the multiprocessing-safe function.
        It can be used for single-threaded evaluation or testing.
        """
        if not _WORKER_CACHE:
            init_worker(
                self.provider_uri,
                self.start_date,
                self.end_date,
                self.benchmark_code,
                self.csi500_membership_path,
            )
        return evaluate_factor_mp(
            factor_expression,
            self.start_date,
            self.end_date
        )


def main():
    parser = argparse.ArgumentParser(description="Evaluate a single factor expression.")
    parser.add_argument("expression", help="Factor expression, e.g., Div($close, Ref($close, 1))")
    parser.add_argument("--start", default="2023-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2024-01-01", help="End date (YYYY-MM-DD)")
    parser.add_argument("--benchmark", default="sh000300", help="Benchmark instrument code")
    parser.add_argument("--provider", default="./qlib_bin_data", help="Qlib provider uri")
    parser.add_argument("--csi500_membership", default=str(DEFAULT_CSI500_MEMBERSHIP), help="CSI500 membership csv path")
    args = parser.parse_args()

    init_worker(args.provider, args.start, args.end, args.benchmark, args.csi500_membership)

    ir, ic, icir, price_corr, _ = evaluate_factor_mp(args.expression, args.start, args.end)
    print(f"Expression: {args.expression}")
    print(f"IR: {ir:.4f}")
    print(f"IC: {ic:.4f}")
    print(f"ICIR: {icir:.4f}")
    print(f"Price Corr: {price_corr:.4f}")


if __name__ == "__main__":
    main()
