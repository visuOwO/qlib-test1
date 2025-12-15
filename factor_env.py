import qlib
import pandas as pd
import numpy as np
import traceback
from qlib.data import D
from qlib.config import C

# This is the new multiprocessing-safe evaluation function
def evaluate_factor_mp(factor_expression: str, start_date: str, end_date: str, benchmark_code: str, provider_uri: str) -> tuple:
    """
    A self-contained, multiprocessing-safe function to evaluate a factor expression.
    It initializes Qlib within the process.
    """
    try:
        # 1. Initialize Qlib for this worker process
        if not C.get("initialized", False):
            qlib.init(provider_uri=provider_uri, region=qlib.constant.REG_CN)

        # 2. Get benchmark return
        benchmark_ret_df = D.features([benchmark_code], ["Ref($close, -1)/$close - 1"], start_date, end_date)
        if not benchmark_ret_df.empty and isinstance(benchmark_ret_df.index, pd.MultiIndex):
            benchmark_ret_df = benchmark_ret_df.droplevel(0)

        # 3. Get target data (future returns)
        target_df = D.features(D.instruments("all"), ["Ref($close, -1)/$close - 1"], start_time=start_date, end_time=end_date, freq="day")
        if target_df.empty: return -1.0, -1.0, 1.0

        # 4. Get factor data
        try:
            factor_df = D.features(
                D.instruments("all"),
                [factor_expression, "$close", "$industry"], 
                start_time=start_date,
                end_time=end_date,
                freq="day"
            )
        except Exception as e:
            # This can happen if the expression is invalid for Qlib
            return -1.0, -1.0, 1.0

        if factor_df.empty: return -1.0, -1.0, 1.0
        
        if 'SH000300' in factor_df.index.get_level_values('instrument'):
            factor_df = factor_df.drop('SH000300', level='instrument')

        # 5. Data merging and cleaning
        target_df.columns = ['target']
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
        # Broad exception to catch any other errors during evaluation
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
            self.end_date,
            self.benchmark_code,
            self.provider_uri
        )
