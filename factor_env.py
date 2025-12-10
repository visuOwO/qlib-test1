import qlib
import pandas as pd
import numpy as np
import traceback
from qlib.data import D
from qlib.config import C

class FactorMiningEnv:
    def __init__(self, start_date, end_date, benchmark_code="sh000300", provider_uri="./qlib_bin_data"):
        # 初始化 Qlib
        if not C.get("initialized", False):
            qlib.init(provider_uri=provider_uri, region=qlib.constant.REG_CN)
            print(f"Qlib initialized with data from: {provider_uri}")

        self.start_date = start_date
        self.end_date = end_date
        self.benchmark_code = benchmark_code
        self.raw_features = [
            "$open", "$high", "$low", "$close", "$volume", 
            "$amount", "$turnover_rate"
        ]
        self.benchmark_ret = self._get_benchmark_return()

    def _get_benchmark_return(self):
        try:
            df = D.features([self.benchmark_code], ["Ref($close, -1)/$close - 1"], self.start_date, self.end_date)
            if not df.empty and isinstance(df.index, pd.MultiIndex):
                df = df.droplevel(0)
            return df
        except Exception as e:
            print(f"Error loading benchmark: {e}")
            return pd.DataFrame()

    def _get_target_data(self, instrument="all", freq="day"):
        return D.features(
            D.instruments(instrument),
            ["Ref($close, -1)/$close - 1"], 
            start_time=self.start_date,
            end_time=self.end_date,
            freq=freq
        )

    def evaluate_factor(self, factor_expression: str) -> tuple:
        try:
            target_df = self._get_target_data()
            if target_df.empty: return -1.0, -1.0, 1.0

            # Fetch Factor AND Close price for correlation check
            factor_df = D.features(
                D.instruments("all"),
                [factor_expression, "$close"],
                start_time=self.start_date,
                end_time=self.end_date,
                freq="day"
            )

            if factor_df.empty: return -1.0, -1.0, 1.0

            target_df.columns = ['target'] 
            # factor_df has 2 columns: factor_expression and $close
            # We merge them.
            merged_df = pd.merge(factor_df, target_df, left_index=True, right_index=True, how='inner')
            merged_df.dropna(inplace=True)
            if merged_df.empty: return -1.0, -1.0, 1.0

            # Calculate Price Correlation (Spearman)
            # High correlation with price (un-normalized) is bad.
            price_corr_series = merged_df.groupby(level='datetime').apply(
                lambda x: x[factor_expression].corr(x['$close'], method='spearman')
            )
            price_corr = price_corr_series.mean()
            if np.isnan(price_corr): price_corr = 1.0

            # IC
            ic_series = merged_df.groupby(level='datetime').apply(
                lambda x: x[factor_expression].corr(x['target'], method='spearman')
            )
            ic = ic_series.mean()
            if np.isnan(ic): ic = -1.0

            # IR (Top 20% Excess Return / Std)
            def get_group_return(df_day):
                try:
                    df_day['group'] = pd.qcut(df_day[factor_expression], 5, labels=False, duplicates='drop')
                    return df_day.groupby('group')['target'].mean()
                except ValueError:
                    return pd.Series()

            daily_group_returns = merged_df.groupby(level='datetime').apply(get_group_return)
            if 4 not in daily_group_returns.columns: return -1.0, ic, price_corr

            long_ret = daily_group_returns[4]
            if self.benchmark_ret.empty:
                bench_ret = pd.Series(0, index=long_ret.index)
            else:
                bench_ret = self.benchmark_ret.reindex(long_ret.index).fillna(0).iloc[:, 0]

            excess_ret = long_ret - bench_ret
            mean_excess = excess_ret.mean()
            std_excess = excess_ret.std()

            if std_excess == 0 or np.isnan(std_excess):
                ir = -1.0
            else:
                ir = (mean_excess / std_excess) * np.sqrt(252)
                if np.isnan(ir): ir = -1.0
            
            return ir, ic, price_corr

        except Exception as e:
            # print(f"Evaluation Error: {e}")
            return -1.0, -1.0, 1.0
