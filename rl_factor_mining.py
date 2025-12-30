import torch
import os
from datetime import datetime
from factor_env import FactorMiningEnv
from rl_agent import DeepQLearningAgent
from factor_visualization import visualize_factor

if __name__ == "__main__":
    # 配置回测时间
    EVAL_START_DATE = "2020-01-01"
    EVAL_END_DATE = "2024-12-31"

    # 检查 PyTorch 是否可用
    if torch.cuda.is_available():
        print("CUDA is available. Training on GPU.")
    else:
        print("CUDA not available. Training on CPU.")

    # 初始化环境
    env = FactorMiningEnv(start_date=EVAL_START_DATE, end_date=EVAL_END_DATE)
    
    # 初始化代理
    agent = DeepQLearningAgent(env, epsilon=1.0)
    
    # 开始训练
    # 建议增加 episodes 数量以获得更好的结果
    result = agent.train(target_valid_episodes=500, max_attempts=10000)
    # 记录最佳因子到文件
    os.makedirs("analysis_results", exist_ok=True)
    summary_path = os.path.join("analysis_results", "best_factor_summary.txt")
    with open(summary_path, "a", encoding="utf-8") as f:
        f.write(
            "\n".join(
                [
                    f"=== {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===",
                    f"Best Factor: {result.get('best_factor', '')}",
                    f"Best IR: {result.get('best_ir', '')}",
                    f"Best IC: {result.get('best_ic', '')}",
                    f"Best ICIR: {result.get('best_icir', '')}",
                    f"Total Attempts: {result.get('total_attempts', '')}",
                    f"Valid Factors: {result.get('valid_count', '')}",
                    f"Train Window: {EVAL_START_DATE} ~ {EVAL_END_DATE}",
                    "",
                ]
            )
        )
    print(f"最佳因子摘要已写入: {summary_path}")

    # 训练完成后可视化最佳因子
    best_expr = result.get("best_factor", "")
    if best_expr:
        print(f"\n开始可视化最佳因子: {best_expr}")
        try:
            visualize_factor(
                expr=best_expr,
                start=EVAL_START_DATE,
                end=EVAL_END_DATE,
                provider=env.provider_uri,
                benchmark=env.benchmark_code,
                groups=5,
                focus_group=4,
            )
        except Exception as e:
            print(f"可视化失败: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("未找到有效的最佳因子，跳过可视化。")
