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
            "$open", "$high", "$low", "$close", "$volume", "$amount", 
            "$turnover_rate", "$turnover_rate_f", "$volume_ratio",
            "$pe_ttm", "$pb", "$ps_ttm", "$dv_ttm",
            "$total_mv", "$circ_mv"
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
            # 1. 获取标签数据 (未来收益率)
            target_df = self._get_target_data()
            if target_df.empty: return -1.0, -1.0, 1.0

            # 2. 获取因子数据、收盘价(用于查重)、行业分类(用于中性化)
            # 注意：请确保您的 Qlib 数据中已经包含了 $industry 字段
            # 如果尚未添加，临时可以用 "$close" 占位以免报错，但就失去了中性化效果
            try:
                factor_df = D.features(
                    D.instruments("all"),
                    [factor_expression, "$close", "$industry"], 
                    start_time=self.start_date,
                    end_time=self.end_date,
                    freq="day"
                )
            except Exception as e:
                print(f"Data Load Error: {e}")
                return -1.0, -1.0, 1.0

            if factor_df.empty: return -1.0, -1.0, 1.0

            # [修改] 2. 剔除指数 (SH000300)
            # 简单粗暴的方法：直接 drop 掉 index 中包含 'SH000300' 的行
            if 'SH000300' in factor_df.index.get_level_values('instrument'):
                 factor_df = factor_df.drop('SH000300', level='instrument')

            # 3. 数据合并与清洗
            target_df.columns = ['target']
            merged_df = pd.merge(factor_df, target_df, left_index=True, right_index=True, how='inner')
            merged_df.dropna(inplace=True)
            if merged_df.empty: return -1.0, -1.0, 1.0

            # ------------------------------------------------------------------
            # [步骤 A] 原始因子检查 (用于防止风格暴露)
            # ------------------------------------------------------------------
            # 计算原始因子与价格的相关性 (Spearman)
            # 如果因子就是股价本身(或其变体)，这里的相关性会很高
            price_corr_series = merged_df.groupby(level='datetime').apply(
                lambda x: x[factor_expression].corr(x['$close'], method='spearman')
            )
            price_corr = price_corr_series.mean()
            if np.isnan(price_corr): price_corr = 1.0

            # ------------------------------------------------------------------
            # [步骤 B] 行业中性化处理 (Industry Neutralization)
            # ------------------------------------------------------------------
            # 定义中性化函数：Z-Score (去均值除以标准差)
            def neutralize_func(df_group):
                # 如果该行业该天股票太少(<2只)，无法计算标准差，返回0
                if len(df_group) < 2: 
                    return pd.Series(0, index=df_group.index)
                return (df_group - df_group.mean()) / (df_group.std() + 1e-9)

            # 按日期(level='datetime')和行业('$industry')分组进行标准化
            # transform 会保持原来的索引结构
            merged_df['neu_factor'] = merged_df.groupby(['datetime', '$industry'])[factor_expression].transform(neutralize_func)
            
            # 对缺失值填充0 (比如某些股票没有行业数据，或者行业内只有它一只)
            merged_df['neu_factor'] = merged_df['neu_factor'].fillna(0)

            # ------------------------------------------------------------------
            # [步骤 C] 计算指标 (使用中性化后的因子 'neu_factor')
            # ------------------------------------------------------------------
            
            # 1. IC (Information Coefficient)
            ic_series = merged_df.groupby(level='datetime').apply(
                lambda x: x['neu_factor'].corr(x['target'], method='spearman')
            )
            ic = ic_series.mean()
            if np.isnan(ic): ic = -1.0

            # 2. IR (Information Ratio) - 基于分组回测
            def get_group_return(df_day):
                try:
                    # 将中性化后的因子分为 5 组
                    df_day['group'] = pd.qcut(df_day['neu_factor'], 5, labels=False, duplicates='drop')
                    return df_day.groupby('group')['target'].mean()
                except ValueError:
                    return pd.Series()

            daily_group_returns = merged_df.groupby(level='datetime').apply(get_group_return)
            
            # 检查是否成功分出了第 4 组 (Top 组)
            if 4 not in daily_group_returns.columns: 
                return -1.0, ic, price_corr

            # 计算多头超额收益 (Long - Benchmark)
            long_ret = daily_group_returns[4]
            
            # 对齐基准指数
            if self.benchmark_ret.empty:
                bench_ret = pd.Series(0, index=long_ret.index)
            else:
                bench_ret = self.benchmark_ret.reindex(long_ret.index).fillna(0).iloc[:, 0]

            excess_ret = long_ret - bench_ret
            mean_excess = excess_ret.mean()
            std_excess = excess_ret.std()

            # 计算年化 IR
            if std_excess == 0 or np.isnan(std_excess):
                ir = -1.0
            else:
                ir = (mean_excess / std_excess) * np.sqrt(252)
                if np.isnan(ir): ir = -1.0
            
            return ir, ic, price_corr

        except Exception as e:
            # print(f"Evaluation Error: {e}") # 调试时可以打开
            return -1.0, -1.0, 1.0