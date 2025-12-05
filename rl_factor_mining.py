import torch
from factor_env import FactorMiningEnv
from rl_agent import DeepQLearningAgent

if __name__ == "__main__":
    # 配置回测时间
    EVAL_START_DATE = "2023-01-01"
    EVAL_END_DATE = "2024-01-01"

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
    agent.train(episodes=100)