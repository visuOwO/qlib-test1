import qlib
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
                [factor_expression, "$close", "$industry"], 
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

        # 7. Industry Neutralization
        def neutralize_func(df_group):
            if len(df_group) < 2: return pd.Series(0, index=df_group.index)
            return (df_group - df_group.mean()) / (df_group.std() + 1e-9)

        merged_df['neu_factor'] = merged_df.groupby(['datetime', '$industry'])[factor_expression].transform(neutralize_func)
        merged_df['neu_factor'] = merged_df['neu_factor'].fillna(0)

        # 8. Calculate Metrics
        ic_series = merged_df.groupby(level='datetime').apply(
            lambda x: x['neu_factor'].corr(x['target'], method='spearman')
        )
        ic = ic_series.mean()
        if np.isnan(ic): ic = -1.0

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
            return -1.0, ic, price_corr

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
        
        return ir, ic, price_corr

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
        return evaluate_factor_mp(
            factor_expression,
            self.start_date,
            self.end_date
        )
