import qlib
import argparse
import pandas as pd
import numpy as np
import traceback
from qlib.data import D
from qlib.config import C

_WORKER_CACHE = {}

def init_worker(provider_uri, start_date, end_date, benchmark_code):
    """
    子进程初始化函数：只运行一次
    负责初始化 Qlib 并加载公共数据到全局变量
    """
    try:
        if not C.get("initialized", False):
            qlib.init(provider_uri=provider_uri, region=qlib.constant.REG_CN)
        
        # 1. 预加载基准收益
        bench_df = D.features([benchmark_code], ["Ref($close, -1)/$close - 1"], start_date, end_date)
        if not bench_df.empty and isinstance(bench_df.index, pd.MultiIndex):
            bench_df = bench_df.droplevel(0)
            
        # 2. 预加载目标收益 (Target)
        target_df = D.features(D.instruments("all"), ["Ref($close, -1)/$close - 1"], start_time=start_date, end_time=end_date, freq="day")
        target_df.columns = ['target']
        
        # 3. 预加载基础行情 (可选，用于计算相关性，防止每次都去读 $close)
        # 这里我们至少可以把 target 和 benchmark 存起来
        _WORKER_CACHE['benchmark_ret'] = bench_df
        _WORKER_CACHE['target_df'] = target_df
        
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
        
        if target_df is None or target_df.empty:
            return -1.0, -1.0, 1.0

        # 4. Get factor data (因子数据必须动态计算)
        try:
            # 注意：这里我们依然需要 $close 和 $industry
            # Qlib 的 D.features 会处理对齐，所以这里重新读取 $close 开销是可以接受的，或者也可以进一步优化缓存
            factor_df = D.features(
                D.instruments("all"),
                [factor_expression, "$close", "$industry", "$circ_mv"], 
                start_time=start_date,
                end_time=end_date,
                freq="day"
            )
        except Exception as e:
            # 公式错误等情况
            return -1.0, -1.0, 1.0

        if factor_df.empty: return -1.0, -1.0, 1.0
        
        if 'SH000300' in factor_df.index.get_level_values('instrument'):
            factor_df = factor_df.drop('SH000300', level='instrument')

        # 5. Merge (使用缓存的 target_df)
        merged_df = pd.merge(factor_df, target_df, left_index=True, right_index=True, how='inner')
        merged_df.dropna(inplace=True)
        
        if merged_df.empty: return -1.0, -1.0, 1.0

        # 6. Price Correlation Check
        price_corr_series = merged_df.groupby(level='datetime').apply(
            lambda x: x[factor_expression].corr(x['$close'], method='spearman')
        )
        price_corr = price_corr_series.mean()
        if np.isnan(price_corr): price_corr = 1.0

        # 7. Circulating Market Value Neutralization
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

        # 9. Calculate Metrics
        ic_series = merged_df.groupby(level='datetime').apply(
            lambda x: x['neu_factor'].corr(x['target'], method='spearman')
        )
        ic_mean = ic_series.mean()
        ic_std = ic_series.std()
        if np.isnan(ic): ic = -1.0

        if ic_std == 0:
            icir = 0
        else:
            icir = ic_mean / ic_std

        t_stat = icir * np.sqrt(len(ic_series))

        def get_group_return(df_day):
            try:
                df_day['group'] = pd.qcut(df_day['neu_factor'], 5, labels=False, duplicates='drop')
                return df_day.groupby('group')['target'].mean()
            except ValueError:
                return pd.Series()
        
        daily_group_returns = merged_df.groupby(level='datetime').apply(get_group_return)
        
        if isinstance(daily_group_returns, pd.Series):
            if isinstance(daily_group_returns.index, pd.MultiIndex):
                # 情况 1: 结果是 MultiIndex Series (datetime, group) -> Unstack 展开为 DataFrame
                daily_group_returns = daily_group_returns.unstack()
            else:
                # 情况 2: 结果是单日数据，Index 只是 group (0, 1, 2...) -> 转为单行 DataFrame
                # 这种情况下 Series 的 name 可能是 datetime，转置后变为 index
                daily_group_returns = daily_group_returns.to_frame().T
        
        # 此时 daily_group_returns 必定是 DataFrame，可以安全访问 .columns
        if 4 not in daily_group_returns.columns: 
            return -1.0, ic_mean, price_corr

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
        
        return ir, ic_mean, price_corr

    except Exception:
        # 建议打印错误堆栈，否则调试困难
        traceback.print_exc() 
        return -1.0, -1.0, 1.0


class FactorMiningEnv:
    def __init__(self, start_date, end_date, benchmark_code="sh000300", provider_uri="./qlib_bin_data"):
        self.start_date = start_date
        self.end_date = end_date
        self.benchmark_code = benchmark_code
        self.provider_uri = provider_uri
        
        # Initialize Qlib in the main process
        if not C.get("initialized", False):
            qlib.init(provider_uri=self.provider_uri, region=qlib.constant.REG_CN)
            print(f"Qlib initialized with data from: {provider_uri}")

        self.raw_features = [
            "$open", "$high", "$low", "$close", "$volume", "$amount", 
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
            init_worker(self.provider_uri, self.start_date, self.end_date, self.benchmark_code)
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
    args = parser.parse_args()

    init_worker(args.provider, args.start, args.end, args.benchmark)

    ir, ic, price_corr = evaluate_factor_mp(args.expression, args.start, args.end)
    print(f"Expression: {args.expression}")
    print(f"IR: {ir:.4f}")
    print(f"IC: {ic:.4f}")
    print(f"Price Corr: {price_corr:.4f}")


if __name__ == "__main__":
    main()
