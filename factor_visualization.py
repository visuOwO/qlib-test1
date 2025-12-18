import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager, rcParams
import qlib
from qlib.data import D
from qlib.config import C


def init_qlib(provider_uri: str):
    if not C.get("initialized", False):
        qlib.init(provider_uri=provider_uri, region=qlib.constant.REG_CN)


def _set_chinese_font():
    # Try to find a font that supports Chinese to avoid glyph warnings
    candidates = ["SimHei", "Noto Sans CJK SC", "Microsoft YaHei", "PingFang SC"]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            rcParams["font.family"] = name
            rcParams["axes.unicode_minus"] = False
            return
    # Fallback: keep default font but still ensure minus sign works
    rcParams["axes.unicode_minus"] = False
    print("提示：未找到中文字体，可能会出现方块/缺字。可安装 SimHei 或 Noto Sans CJK。")

def prepare_factor_data(expr: str, start: str, end: str, benchmark: str) -> pd.DataFrame:
    # Pull factor, industry, size and next-day return
    factor_df = D.features(
        D.instruments("all"),
        [expr, "$industry", "$circ_mv", "Ref($close, -1)/$close - 1"],
        start_time=start,
        end_time=end,
        freq="day",
    ).copy()

    factor_df.columns = ["factor", "$industry", "$circ_mv", "target"]

    # Remove benchmark row if it leaks into instruments
    if benchmark.upper() in factor_df.index.get_level_values("instrument"):
        factor_df = factor_df.drop(benchmark.upper(), level="instrument")

    factor_df = factor_df.dropna()
    return factor_df


def neutralize_factor(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["log_mkt_cap"] = np.log(df["$circ_mv"] + 1)

    def clip_outliers(series: pd.Series) -> pd.Series:
        mean = series.mean()
        std = series.std()
        return series.clip(mean - 3 * std, mean + 3 * std)

    df["raw_factor"] = df.groupby(level="datetime")["factor"].transform(clip_outliers).fillna(0)

    def regress_out_size(df_day: pd.DataFrame) -> pd.Series:
        y = df_day["raw_factor"].values
        x = df_day["log_mkt_cap"].values
        mask = ~np.isnan(y) & ~np.isnan(x)
        if np.sum(mask) < 10:
            return pd.Series(0, index=df_day.index)

        x_valid = x[mask]
        y_valid = y[mask]
        x_mean = x_valid.mean()
        y_mean = y_valid.mean()
        numerator = np.sum((x_valid - x_mean) * (y_valid - y_mean))
        denominator = np.sum((x_valid - x_mean) ** 2)
        slope = 0 if denominator == 0 else numerator / denominator
        intercept = y_mean - slope * x_mean
        resid = y - (slope * x + intercept)
        return pd.Series(resid, index=df_day.index)

    df["size_neu_factor"] = df.groupby(level="datetime", group_keys=False).apply(regress_out_size)

    def industry_standardize(series: pd.Series) -> pd.Series:
        return (series - series.mean()) / (series.std() + 1e-9)

    df["neu_factor"] = (
        df.groupby(["datetime", "$industry"])["size_neu_factor"].transform(industry_standardize).fillna(0)
    )
    return df


def calc_group_returns(df: pd.DataFrame, group_num: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()

    def assign_group(df_day: pd.DataFrame) -> pd.Series:
        try:
            return pd.qcut(df_day["neu_factor"], group_num, labels=False, duplicates="drop")
        except ValueError:
            return pd.Series(np.nan, index=df_day.index)

    df["group"] = df.groupby(level="datetime", group_keys=False).apply(assign_group)
    df = df.dropna(subset=["group"])
    df["group"] = df["group"].astype(int)

    group_ret = (
        df.groupby(["datetime", "group"])["target"].mean().unstack().sort_index()
    )
    return df, group_ret


def calc_turnover(df: pd.DataFrame, focus_group: int) -> pd.Series:
    turnover_list = []
    prev_members = None
    dates = []

    for date, df_day in df.groupby(level="datetime"):
        dates.append(date)
        members = set(df_day[df_day["group"] == focus_group].index.get_level_values("instrument"))
        if prev_members is None or len(prev_members) == 0:
            turnover_list.append(np.nan)
        else:
            overlap = len(prev_members & members)
            denom = len(prev_members) if len(prev_members) > 0 else 1
            turnover_list.append(1 - overlap / denom)
        prev_members = members

    return pd.Series(turnover_list, index=pd.to_datetime(dates)).sort_index()


def visualize_factor(expr: str, start: str, end: str, provider: str, benchmark: str = "sh000300", groups: int = 5, focus_group: int = 4):
    _set_chinese_font()
    init_qlib(provider)
    df = prepare_factor_data(expr, start, end, benchmark)
    df = neutralize_factor(df)
    df, group_ret = calc_group_returns(df, groups)

    if focus_group not in group_ret.columns:
        raise ValueError(f"目标分组 {focus_group} 不存在，检查 group_num 或因子分布。")

    turnover = calc_turnover(df, focus_group)

    bench_df = D.features([benchmark], ["Ref($close, -1)/$close - 1"], start, end, freq="day")
    if isinstance(bench_df.index, pd.MultiIndex):
        bench_df = bench_df.droplevel(0)
    bench_ret = bench_df.iloc[:, 0]
    bench_ret.index = pd.to_datetime(bench_ret.index)
    bench_index = bench_ret.index

    # 保证基准线日期一致：统一用基准的交易日做索引
    group_ret = group_ret.reindex(bench_index).fillna(0)
    turnover = turnover.reindex(bench_index)
    long_ret = group_ret[focus_group].fillna(0)

    os.makedirs("analysis_results", exist_ok=True)

    fig, axes = plt.subplots(4, 1, figsize=(14, 16), sharex=True)

    (1 + group_ret).cumprod().plot(ax=axes[0])
    axes[0].set_title(f"{expr} 分层累计收益 (分组数={groups})")
    axes[0].set_ylabel("Net Value")
    axes[0].grid(True)

    focus_cum = (1 + long_ret).cumprod()
    bench_cum = (1 + bench_ret).cumprod()
    axes[1].plot(focus_cum.index, focus_cum, label=f"Group {focus_group} 累计收益")
    axes[1].plot(bench_cum.index, bench_cum, label=f"{benchmark} 累计收益", alpha=0.7)
    axes[1].set_title(f"组 {focus_group} 累计收益 vs 基准")
    axes[1].set_ylabel("Net Value")
    axes[1].legend()
    axes[1].grid(True)

    turnover.plot(ax=axes[2], color="tab:orange")
    axes[2].set_title(f"组 {focus_group} 每日换手率")
    axes[2].set_ylabel("Turnover")
    axes[2].grid(True)

    axes[3].plot(long_ret.index, long_ret, label=f"Group {focus_group} 日收益")
    axes[3].plot(bench_ret.index, bench_ret, label=f"{benchmark} 日收益", alpha=0.7)
    axes[3].set_title("每日收益对比")
    axes[3].set_ylabel("Return")
    axes[3].legend()
    axes[3].grid(True)

    fig.autofmt_xdate()
    save_path = os.path.join("analysis_results", "factor_visualization.png")
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"图表已保存到: {save_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="可视化因子的分层收益、换手率和每日收益。")
    parser.add_argument("expression", help='因子表达式，例如 Div($close, Ref($close, 1))')
    parser.add_argument("--start", default="2023-01-01", help="开始日期 YYYY-MM-DD")
    parser.add_argument("--end", default="2024-01-01", help="结束日期 YYYY-MM-DD")
    parser.add_argument("--provider", default="./qlib_bin_data", help="Qlib 数据目录")
    parser.add_argument("--benchmark", default="sh000300", help="基准代码，例如 sh000300")
    parser.add_argument("--groups", type=int, default=5, help="分层数量")
    parser.add_argument("--focus_group", type=int, default=4, help="关注的分组编号（0-index）")
    args = parser.parse_args()

    visualize_factor(
        expr=args.expression,
        start=args.start,
        end=args.end,
        provider=args.provider,
        benchmark=args.benchmark,
        groups=args.groups,
        focus_group=args.focus_group,
    )


if __name__ == "__main__":
    main()
