import numpy as np
import pandas as pd
import qlib
from qlib.config import C
from qlib.data import D

from factor_env import (
    _align_membership_index,
    _load_csi500_membership,
    DEFAULT_CSI500_MEMBERSHIP,
    expand_vwap_expressions,
)


def _ensure_qlib_initialized(provider_uri: str):
    if not C.get("initialized", False):
        qlib.init(provider_uri=provider_uri, region=qlib.constant.REG_CN)


def _clip_outliers(series: pd.Series) -> pd.Series:
    mean = series.mean()
    std = series.std()
    if np.isnan(std) or std == 0:
        return series
    return series.clip(mean - 3 * std, mean + 3 * std)


def _regress_out_size(df_day: pd.DataFrame, factor_col: str) -> pd.Series:
    y = df_day[factor_col].values
    x = df_day["log_mkt_cap"].values

    valid_mask = ~np.isnan(y) & ~np.isnan(x)
    if np.sum(valid_mask) < 10:
        return pd.Series(0.0, index=df_day.index)

    y_valid = y[valid_mask]
    x_valid = x[valid_mask]

    x_mean = np.mean(x_valid)
    y_mean = np.mean(y_valid)
    numerator = np.sum((x_valid - x_mean) * (y_valid - y_mean))
    denominator = np.sum((x_valid - x_mean) ** 2)
    slope = numerator / denominator if denominator != 0 else 0.0
    intercept = y_mean - slope * x_mean

    resid = y - (slope * x + intercept)
    return pd.Series(resid, index=df_day.index)


def _industry_neutralize_and_standardize(series: pd.Series) -> pd.Series:
    return (series - series.mean()) / (series.std() + 1e-9)


class LinearModelFitter:
    """
    线性模型拟合器，在初始化时缓存数据以加速 fit_linear_ic 调用。
    
    特性：
    - 初始化时缓存 target、$industry、$circ_mv 数据
    - 缓存 CSI500 成员索引
    - fit_linear_ic 只需加载因子数据，复用缓存
    """
    
    def __init__(
        self,
        start_date: str,
        end_date: str,
        provider_uri: str = "./qlib_bin_data",
        csi500_membership_path=DEFAULT_CSI500_MEMBERSHIP,
    ):
        self.start_date = start_date
        self.end_date = end_date
        self.provider_uri = provider_uri
        self.csi500_membership_path = csi500_membership_path
        
        _ensure_qlib_initialized(provider_uri)
        
        # 预加载并缓存数据
        self._preload_data()
    
    def _preload_data(self):
        """预加载 target、industry、circ_mv 等数据"""
        print(f"[LinearModelFitter] Preloading data from {self.start_date} to {self.end_date}...")
        
        # 加载目标收益
        self.target_df = D.features(
            D.instruments("all"),
            ["Ref($close, -5) / Ref($close, -1) - 1"],
            start_time=self.start_date,
            end_time=self.end_date,
            freq="day",
        )
        self.target_df.columns = ["target"]
        
        # 加载基础数据（行业、市值）
        self.base_df = D.features(
            D.instruments("all"),
            ["$industry", "$circ_mv"],
            start_time=self.start_date,
            end_time=self.end_date,
            freq="day",
        )
        
        # 过滤 SH000300
        if "SH000300" in self.target_df.index.get_level_values("instrument"):
            self.target_df = self.target_df.drop("SH000300", level="instrument")
        if "SH000300" in self.base_df.index.get_level_values("instrument"):
            self.base_df = self.base_df.drop("SH000300", level="instrument")
        
        # 预计算 log_mkt_cap
        self.base_df["log_mkt_cap"] = np.log(self.base_df["$circ_mv"] + 1)
        
        # 加载 CSI500 成员索引
        self.csi500_index = _load_csi500_membership(
            self.csi500_membership_path or DEFAULT_CSI500_MEMBERSHIP
        )
        
        print(f"[LinearModelFitter] Preloaded: target={self.target_df.shape}, base={self.base_df.shape}")
    
    def fit_linear_ic(self, factor_expressions):
        """
        使用缓存数据拟合线性模型，返回 (ic_mean, weights_dict)。
        如果拟合失败，返回 (None, {})。
        """
        if not factor_expressions:
            return 0.0, {}
        
        factor_expressions = list(factor_expressions)
        expanded_exprs, rename_map = expand_vwap_expressions(factor_expressions)
        factor_name_map = {expr: f"factor_{i}" for i, expr in enumerate(factor_expressions)}
        
        # 只加载因子数据
        try:
            factor_df = D.features(
                D.instruments("all"),
                expanded_exprs,
                start_time=self.start_date,
                end_time=self.end_date,
                freq="day",
            )
        except Exception:
            return None, {}
        
        if factor_df.empty:
            return None, {}
        
        # 过滤 SH000300
        if "SH000300" in factor_df.index.get_level_values("instrument"):
            factor_df = factor_df.drop("SH000300", level="instrument")
        
        if rename_map:
            factor_df = factor_df.rename(columns=rename_map)
        factor_df = factor_df.rename(columns=factor_name_map)
        
        # 与缓存数据合并
        merged_df = pd.merge(
            factor_df, self.base_df,
            left_index=True, right_index=True, how="inner"
        )
        merged_df = pd.merge(
            merged_df, self.target_df,
            left_index=True, right_index=True, how="inner"
        )
        merged_df.dropna(inplace=True)
        
        if merged_df.empty:
            return None, {}
        
        # 中性化处理
        neu_cols = []
        for expr, col in factor_name_map.items():
            raw_col = f"{col}_raw"
            size_col = f"{col}_size_neu"
            neu_col = f"{col}_neu"
            
            merged_df[raw_col] = merged_df.groupby(level="datetime")[col].transform(_clip_outliers)
            merged_df[raw_col] = merged_df[raw_col].fillna(0)
            
            merged_df[size_col] = merged_df.groupby(level="datetime", group_keys=False).apply(
                lambda df_day, rc=raw_col: _regress_out_size(df_day, rc)
            )
            
            merged_df[neu_col] = merged_df.groupby(["datetime", "$industry"])[size_col].transform(
                _industry_neutralize_and_standardize
            )
            merged_df[neu_col] = merged_df[neu_col].fillna(0)
            neu_cols.append(neu_col)
        
        # CSI500 过滤
        analysis_df = merged_df
        if self.csi500_index is not None:
            in_membership = _align_membership_index(analysis_df.index, self.csi500_index)
            analysis_df = analysis_df.loc[in_membership.values]
        
        if analysis_df.empty:
            return None, {}
        
        # 最小二乘拟合
        x = analysis_df[neu_cols].values
        y = analysis_df["target"].values
        if x.size == 0 or y.size == 0:
            return None, {}
        
        try:
            weights, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
        except Exception:
            return None, {}
        
        # 计算 IC
        preds = x.dot(weights)
        pred_series = pd.Series(preds, index=analysis_df.index, name="pred")
        ic_series = analysis_df.assign(pred=pred_series).groupby(level="datetime").apply(
            lambda df_day: df_day["pred"].corr(df_day["target"], method="spearman")
        )
        ic_mean = ic_series.mean()
        if np.isnan(ic_mean):
            ic_mean = -1.0
        
        weights_dict = {
            expr: float(weights[i]) for i, expr in enumerate(factor_expressions)
        }
        return ic_mean, weights_dict


def fit_linear_ic(
    factor_expressions,
    start_date: str,
    end_date: str,
    provider_uri: str = "./qlib_bin_data",
    csi500_membership_path=DEFAULT_CSI500_MEMBERSHIP,
):
    """
    Fit a linear model on factor signals to predict target returns and return IC + weights.
    Returns (ic_mean, weights_dict). If fitting fails, returns (None, {}).
    """
    if not factor_expressions:
        return 0.0, {}

    _ensure_qlib_initialized(provider_uri)

    factor_expressions = list(factor_expressions)
    expanded_exprs, rename_map = expand_vwap_expressions(factor_expressions)
    factor_name_map = {expr: f"factor_{i}" for i, expr in enumerate(factor_expressions)}

    try:
        target_df = D.features(
            D.instruments("all"),
            ["Ref($close, -5) / Ref($close, -1) - 1"],
            start_time=start_date,
            end_time=end_date,
            freq="day",
        )
        target_df.columns = ["target"]

        factor_df = D.features(
            D.instruments("all"),
            expanded_exprs + ["$industry", "$circ_mv"],
            start_time=start_date,
            end_time=end_date,
            freq="day",
        )
    except Exception:
        return None, {}

    if factor_df.empty or target_df.empty:
        return None, {}

    if "SH000300" in factor_df.index.get_level_values("instrument"):
        factor_df = factor_df.drop("SH000300", level="instrument")

    if rename_map:
        factor_df = factor_df.rename(columns=rename_map)
    factor_df = factor_df.rename(columns=factor_name_map)

    merged_df = pd.merge(factor_df, target_df, left_index=True, right_index=True, how="inner")
    merged_df.dropna(inplace=True)
    if merged_df.empty:
        return None, {}

    merged_df["log_mkt_cap"] = np.log(merged_df["$circ_mv"] + 1)

    neu_cols = []
    for expr, col in factor_name_map.items():
        raw_col = f"{col}_raw"
        size_col = f"{col}_size_neu"
        neu_col = f"{col}_neu"

        merged_df[raw_col] = merged_df.groupby(level="datetime")[col].transform(_clip_outliers)
        merged_df[raw_col] = merged_df[raw_col].fillna(0)

        merged_df[size_col] = merged_df.groupby(level="datetime", group_keys=False).apply(
            lambda df_day: _regress_out_size(df_day, raw_col)
        )

        merged_df[neu_col] = merged_df.groupby(["datetime", "$industry"])[size_col].transform(
            _industry_neutralize_and_standardize
        )
        merged_df[neu_col] = merged_df[neu_col].fillna(0)
        neu_cols.append(neu_col)

    analysis_df = merged_df
    csi500_index = _load_csi500_membership(csi500_membership_path or DEFAULT_CSI500_MEMBERSHIP)
    if csi500_index is not None:
        in_membership = _align_membership_index(analysis_df.index, csi500_index)
        analysis_df = analysis_df.loc[in_membership.values]
    if analysis_df.empty:
        return None, {}

    x = analysis_df[neu_cols].values
    y = analysis_df["target"].values
    if x.size == 0 or y.size == 0:
        return None, {}

    try:
        weights, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
    except Exception:
        return None, {}

    preds = x.dot(weights)
    pred_series = pd.Series(preds, index=analysis_df.index, name="pred")
    ic_series = analysis_df.assign(pred=pred_series).groupby(level="datetime").apply(
        lambda df_day: df_day["pred"].corr(df_day["target"], method="spearman")
    )
    ic_mean = ic_series.mean()
    if np.isnan(ic_mean):
        ic_mean = -1.0

    weights_dict = {
        expr: float(weights[i]) for i, expr in enumerate(factor_expressions)
    }
    return ic_mean, weights_dict


def evaluate_factor_quality(
    factor_expression: str,
    start_date: str,
    end_date: str,
    provider_uri: str = "./qlib_bin_data",
    csi500_membership_path=DEFAULT_CSI500_MEMBERSHIP,
):
    """
    Evaluate a factor for correlation and monotonicity without computing IC/IR.
    Returns (price_corr, monotonicity) or (None, None) on failure.
    """
    if not factor_expression:
        return None, None

    _ensure_qlib_initialized(provider_uri)

    expanded_exprs, rename_map = expand_vwap_expressions([factor_expression])
    try:
        target_df = D.features(
            D.instruments("all"),
            ["Ref($close, -5) / Ref($close, -1) - 1"],
            start_time=start_date,
            end_time=end_date,
            freq="day",
        )
        target_df.columns = ["target"]

        factor_df = D.features(
            D.instruments("all"),
            expanded_exprs + ["$close", "$industry", "$circ_mv"],
            start_time=start_date,
            end_time=end_date,
            freq="day",
        )
    except Exception as e:
        print(f"[Quality Eval Error] expr={factor_expression} error={e}")
        return None, None

    if factor_df.empty or target_df.empty:
        print(f"[Quality Eval Empty] expr={factor_expression} factor_df={factor_df.shape} target_df={target_df.shape}")
        return None, None

    if "SH000300" in factor_df.index.get_level_values("instrument"):
        factor_df = factor_df.drop("SH000300", level="instrument")

    if rename_map:
        factor_df = factor_df.rename(columns=rename_map)
    merged_df = pd.merge(factor_df, target_df, left_index=True, right_index=True, how="inner")
    merged_df.dropna(inplace=True)
    if merged_df.empty:
        print(f"[Quality Eval Empty] expr={factor_expression} merged_df is empty after merge")
        return None, None

    merged_df["log_mkt_cap"] = np.log(merged_df["$circ_mv"] + 1)

    merged_df["raw_factor"] = merged_df.groupby(level="datetime")[factor_expression].transform(_clip_outliers)
    merged_df["raw_factor"] = merged_df["raw_factor"].fillna(0)

    merged_df["size_neu_factor"] = merged_df.groupby(level="datetime", group_keys=False).apply(
        lambda df_day: _regress_out_size(df_day, "raw_factor")
    )

    merged_df["neu_factor"] = merged_df.groupby(["datetime", "$industry"])["size_neu_factor"].transform(
        _industry_neutralize_and_standardize
    )
    merged_df["neu_factor"] = merged_df["neu_factor"].fillna(0)

    analysis_df = merged_df
    csi500_index = _load_csi500_membership(csi500_membership_path or DEFAULT_CSI500_MEMBERSHIP)
    if csi500_index is not None:
        in_membership = _align_membership_index(analysis_df.index, csi500_index)
        analysis_df = analysis_df.loc[in_membership.values]
    if analysis_df.empty:
        print(f"[Quality Eval Empty] expr={factor_expression} analysis_df is empty after membership filter")
        return None, None

    price_corr_series = analysis_df.groupby(level="datetime").apply(
        lambda x: x[factor_expression].corr(x["$close"], method="spearman")
    )
    price_corr = price_corr_series.mean()
    if np.isnan(price_corr):
        price_corr = 1.0

    def get_group_return(df_day):
        try:
            df_day["group"] = pd.qcut(df_day["neu_factor"], 5, labels=False, duplicates="drop")
            return df_day.groupby("group")["target"].mean()
        except ValueError:
            return pd.Series()

    daily_group_returns = analysis_df.groupby(level="datetime").apply(get_group_return)
    if isinstance(daily_group_returns, pd.Series):
        if isinstance(daily_group_returns.index, pd.MultiIndex):
            daily_group_returns = daily_group_returns.unstack()
        else:
            daily_group_returns = daily_group_returns.to_frame().T

    daily_group_returns = daily_group_returns.reindex(columns=range(5))
    if 4 not in daily_group_returns.columns:
        return price_corr, 0.0

    def daily_monotone(row):
        if row.isna().any():
            return np.nan
        return row.corr(pd.Series(range(5)), method="spearman")

    mono_series = daily_group_returns.apply(daily_monotone, axis=1)
    monotonicity = mono_series.dropna().mean() if not mono_series.empty else 0.0
    if np.isnan(monotonicity):
        monotonicity = 0.0

    return price_corr, monotonicity
