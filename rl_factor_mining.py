import torch
import os
from datetime import datetime
from factor_env import FactorMiningEnv
from rl_agent import DeepQLearningAgent
from factor_visualization import visualize_linear_model

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
    top_factors = result.get("top_factors", [])
    best_expr = top_factors[0][0] if top_factors else ""
    with open(summary_path, "a", encoding="utf-8") as f:
        f.write(
            "\n".join(
                [
                    f"=== {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===",
                    f"Best Factor: {best_expr}",
                    f"Total Attempts: {result.get('total_attempts', '')}",
                    f"Valid Factors: {result.get('valid_count', '')}",
                    f"Train Window: {EVAL_START_DATE} ~ {EVAL_END_DATE}",
                    f"Top Factors: {len(top_factors)}",
                    "",
                ]
            )
        )
        if top_factors:
            for i, (expr, ir, ic, icir, mono, weight) in enumerate(top_factors, 1):
                f.write(
                    f"{i}. Weight={weight:.6f} | IR={ir:.4f} | IC={ic:.4f} | "
                    f"ICIR={icir:.4f} | Mono={mono:.4f}\n"
                )
                f.write(f"   {expr}\n\n")
    print(f"最佳因子摘要已写入: {summary_path}")

    # 训练完成后可视化线性模型
    if top_factors:
        factor_exprs = [expr for expr, *_ in top_factors]
        print(f"\n开始可视化线性模型 (因子数: {len(factor_exprs)})")
        try:
            visualize_linear_model(
                expressions=factor_exprs,
                fit_start=EVAL_START_DATE,
                fit_end=EVAL_END_DATE,
                start=EVAL_START_DATE,
                end=EVAL_END_DATE,
                provider=env.provider_uri,
                benchmark=env.benchmark_code,
                groups=5,
                focus_group=4,
                max_factors=agent.top_k,
            )
        except Exception as e:
            print(f"可视化失败: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("未找到有效的因子，跳过可视化。")
