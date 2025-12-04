import qlib
import pandas as pd
import numpy as np
from qlib.data import D
from qlib.utils import init_instance_by_config
from qlib.config import C

# --- 1. Qlib 初始化 ---
# 请确保您的 Qlib 数据已准备好 (通过 dump_bin.py 生成)
PROVIDER_URI = "./qlib_bin_data"
if not C.get("initialized", False):
    qlib.init(provider_uri=PROVIDER_URI, region=qlib.constant.REG_CN)
    print(f"Qlib initialized with data from: {PROVIDER_URI}")

# --- 2. 因子评估环境 (简化版) ---
class FactorMiningEnv:
    def __init__(self, start_date, end_date, benchmark_code="SH000300"):
        self.start_date = start_date
        self.end_date = end_date
        self.benchmark_code = benchmark_code
        self.raw_features = ["$open", "$high", "$low", "$close", "$volume"] # 基础因子，可以扩展

    def _get_target_data(self, instrument="all", freq="day"):
        # 获取未来1日收益作为标签
        # Ref($close, -1) / $close - 1 是未来1日收益率
        # 为了简化，这里直接用 Qlib 提供的标签数据
        # (实际中，您可能需要自己定义标签，比如未来5日收益等)
        return D.features(
            D.instruments(instrument),
            ["Ref($close, -1)/$close - 1"], # 未来1日收益作为标签
            start_time=self.start_date,
            end_time=self.end_date,
            freq=freq
        )

    def evaluate_factor(self, factor_expression: str) -> tuple:
        """
        评估一个因子表达式的表现，返回 (Sharpe Ratio, IC)。
        """
        try:
            # 1. 获取目标 (标签) 数据
            target_df = self._get_target_data()
            if target_df.empty:
                print("Warning: Target data is empty.")
                return -1.0, -1.0

            # 2. 获取自定义因子数据
            factor_df = D.features(
                D.instruments("all"),
                [factor_expression],
                start_time=self.start_date,
                end_time=self.end_date,
                freq="day"
            )

            if factor_df.empty:
                print(f"Warning: Factor data for '{factor_expression}' is empty.")
                return -1.0, -1.0

            # 3. 合并数据
            target_df.columns = ['target'] 
            merged_df = pd.merge(
                factor_df, 
                target_df, 
                left_index=True, 
                right_index=True,
                how='inner'
            )
            
            merged_df.dropna(inplace=True)

            if merged_df.empty:
                return -1.0, -1.0

            # --- 计算 IC ---
            ic_series = merged_df.groupby(level='datetime').apply(
                lambda x: x[factor_expression].corr(x['target'], method='spearman')
            )
            ic = ic_series.mean()
            if np.isnan(ic): ic = -1.0

            # --- 计算 Sharpe Ratio ---
            def get_group_return(df_day):
                try:
                    # 将当天所有股票按因子值分为 5 组
                    df_day['group'] = pd.qcut(df_day[factor_expression], 5, labels=False, duplicates='drop')
                    # 计算每组的平均收益
                    return df_day.groupby('group')['target'].mean()
                except ValueError:
                    return pd.Series()

            daily_group_returns = merged_df.groupby(level='datetime').apply(get_group_return)
            
            if 4 not in daily_group_returns.columns or 0 not in daily_group_returns.columns:
                 return -1.0, ic

            long_ret = daily_group_returns[4]
            short_ret = daily_group_returns[0]
            strategy_ret = long_ret - short_ret

            mean_ret = strategy_ret.mean()
            std_ret = strategy_ret.std()

            if std_ret == 0 or np.isnan(std_ret):
                sharpe = -1.0
            else:
                sharpe = (mean_ret / std_ret) * np.sqrt(252)
                if np.isnan(sharpe): sharpe = -1.0
            
            return sharpe, ic

        except Exception as e:
            # print(f"Error evaluating factor '{factor_expression}': {e}")
            return -1.0, -1.0

# --- 3. 简单的 RL Agent (随机 Agent) ---
class RandomFactorAgent:
    def __init__(self, env: FactorMiningEnv):
        self.env = env
        self.operators = ["Add", "Sub", "Mul", "Div", "Mean", "Ref"] # 可用的操作符
        self.operands = env.raw_features # 可用的操作数 (原始因子)

    def _get_random_operand(self):
        return np.random.choice(self.operands)

    def _get_random_operator(self):
        return np.random.choice(self.operators)

    def generate_factor_expression(self, max_depth=2) -> str:
        """
        随机生成一个简单因子表达式。
        为了简化，这里只生成形如 Op(Operand1, Operand2) 或 Op(Operand, N) 的表达式
        """
        operator = self._get_random_operator()

        if operator in ["Add", "Sub", "Mul", "Div"]:
            # 二元操作符
            op1 = self._get_random_operand()
            op2 = self._get_random_operand()
            return f"{operator}({op1}, {op2})"
        elif operator == "Mean":
            # 均值操作，需要一个窗口期
            op = self._get_random_operand()
            window = np.random.randint(5, 30) # 随机选择5到30天窗口
            return f"{operator}({op}, {window})"
        elif operator == "Ref":
            # 引用操作，需要一个偏移量
            op = self._get_random_operand()
            offset = np.random.randint(1, 10) # 随机选择1到10天偏移
            return f"{operator}({op}, {offset})"
        
        return self._get_random_operand() # 作为兜底，直接返回一个原始因子

    def run(self, num_iterations=10):
        print("\n--- 开始强化学习因子挖掘 (随机 Agent 演示) ---")
        best_factor = ""
        best_sharpe = -float('inf')
        best_ic_for_sharpe = 0.0

        for i in range(num_iterations):
            print(f"\n--- 迭代 {i+1}/{num_iterations} ---")
            factor_expr = self.generate_factor_expression()
            print(f"生成的因子表达式: {factor_expr}")

            sharpe, ic = self.env.evaluate_factor(factor_expr)
            print(f"评估结果: Sharpe={sharpe:.4f}, IC={ic:.4f}")

            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_ic_for_sharpe = ic
                best_factor = factor_expr
                print(f"!!! 发现更好的因子: {best_factor} (Sharpe: {best_sharpe:.4f}, IC: {best_ic_for_sharpe:.4f}) !!!")
            
        print("\n--- 因子挖掘完成 ---")
        print(f"最佳因子表达式: {best_factor}")
        print(f"最佳 Sharpe Ratio: {best_sharpe:.4f}")
        print(f"对应的 IC Score: {best_ic_for_sharpe:.4f}")

# --- 主程序入口 ---
if __name__ == "__main__":
    # 配置回测时间，请确保这段时间内的 Qlib 数据是完整的
    EVAL_START_DATE = "2023-01-01"
    EVAL_END_DATE = "2024-01-01"

    env = FactorMiningEnv(start_date=EVAL_START_DATE, end_date=EVAL_END_DATE)
    agent = RandomFactorAgent(env)
    agent.run(num_iterations=20) # 运行20次迭代来生成和评估因子
