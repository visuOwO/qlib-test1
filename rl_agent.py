import torch
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
import collections
from factor_builder import FactorBuilder
from factor_validator import FactorValidator
from dqn_model import DQN, RNN_DQN

class DeepQLearningAgent:
    def __init__(self, env, hidden_dim=128, lr=1e-3, gamma=0.99, epsilon=1.0):
        self.env = env
        self.builder = FactorBuilder(max_depth=3, features=env.raw_features)
        
        self.action_dim = len(self.builder.action_map)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Agent using device: {self.device}")
        
        self.policy_net = RNN_DQN(action_dim=self.action_dim, hidden_dim=hidden_dim).to(self.device)
        self.target_net = RNN_DQN(action_dim=self.action_dim, hidden_dim=hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = collections.deque(maxlen=2000)
        self.validator = FactorValidator()
        
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.batch_size = 64

    def select_action(self, state, valid_actions):
        if random.random() < self.epsilon:
            return random.choice(valid_actions)
        
        with torch.no_grad():
            state_t = torch.LongTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_t)
            
            # Mask invalid actions with -inf
            full_mask = torch.full(q_values.shape, -float('inf')).to(self.device)
            full_mask[0, valid_actions] = 0 # Unmask valid
            
            masked_q = q_values + full_mask
            return masked_q.argmax().item()

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)

        state_batch = torch.LongTensor(np.array(state_batch)).to(self.device)
        action_batch = torch.LongTensor(action_batch).unsqueeze(1).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).unsqueeze(1).to(self.device)
        next_state_batch = torch.LongTensor(np.array(next_state_batch)).to(self.device)
        done_batch = torch.FloatTensor(done_batch).unsqueeze(1).to(self.device)

        # Compute Q(s, a)
        q_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        with torch.no_grad():
            next_state_values = self.target_net(next_state_batch).max(1)[0].unsqueeze(1)
        
        # Compute expected Q
        expected_q_values = reward_batch + (self.gamma * next_state_values * (1 - done_batch))

        loss = F.mse_loss(q_values, expected_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self, target_valid_episodes=50, max_attempts=1000):
        """
        target_valid_episodes: 希望生成的【合法】因子数量
        max_attempts: 最大尝试次数（防止模型太笨一直死循环）
        """
        print(f"\n--- Starting Deep Q-Learning (Target: {target_valid_episodes} Valid Factors) ---")
        best_ir = -float('inf')
        best_ic = -float('inf') # Track best IC as well
        best_factor = ""

        valid_count = 0    # 记录生成了多少个合法因子
        total_attempts = 0 # 记录总共尝试了多少次

        # 用于记录历史生成过的公式
        seen_factors = set()
        
        while valid_count < target_valid_episodes and total_attempts < max_attempts:
            state = self.builder.reset()
            total_attempts += 1

            # 临时变量
            episode_reward = 0
            done = False
            
            while not done:
                valid_actions = self.builder.get_valid_actions()
                action = self.select_action(state, valid_actions)
                next_state, done = self.builder.step(action)
                
                # Intermediate reward is 0, final reward depends on factor evaluation
                reward = 0

                if done:
                    expr = self.builder.build_expression()
                    # print(f"Evaluating: {expr}")

                    # 1. 语法校验
                    if not self.validator.validate(expr):
                        # 如果物理量纲错误 (如 Price + Volume)
                        # print(f"Invalid Logic (Type Mismatch): {expr}")
                        reward = -10.0 # 给一个较重的惩罚，告诉它不要这样做

                    # 2. 重复性校验
                    elif expr in seen_factors:
                        # 如果生成了重复的因子（比如第100次生成 $turnover_rate）
                        # 给予微小的惩罚，告诉它“这个我要过了，换个新的”
                        # print(f"Duplicate Factor: {expr}")
                        reward = -5.0

                    else:
                        # 这是一个全新的、合法的因子
                        seen_factors.add(expr) # 加入已见集合
                        print(f"Attempt {total_attempts} (Valid #{valid_count+1}): {expr}")
                        
                        # 3. 只有全新的因子才去跑回测
                        ir, ic, price_corr = self.env.evaluate_factor(expr)
                        
                        # Penalize high correlation with raw price (un-normalized factors)
                        is_correlated = abs(price_corr) > 0.6

                        if ir > best_ir and not is_correlated:
                            best_ir = ir
                            best_ic = ic # Update best IC
                            best_factor = expr
                            print(f"New Best! IR: {best_ir:.4f} | IC: {best_ic:.4f} | Corr: {price_corr:.4f} | {expr}")

                        # Reward Engineering
                        if is_correlated:
                            reward = -5.0 # Heavy penalty for just mimicking price
                        elif ir > 0:
                            print(f"Valid & New: {expr} | IC: {ic:.4f}")
                            valid_count += 1
                            reward = ir * 20 # Amplify positive IR
                        else:
                            reward = -1.0 # Penalty for bad factor or error
                
                self.memory.append((state, action, reward, next_state, done))
                state = next_state
                episode_reward += reward
                
                self.optimize_model()

            # Epsilon Decay (可以按尝试次数衰减，也可以按有效次数衰减，这里按尝试次数)
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            if total_attempts % 10 == 0:
                print(f"Progress: {valid_count}/{target_valid_episodes} Valid Factors | Total Attempts: {total_attempts} | Epsilon: {self.epsilon:.2f}")

        print("\n--- Training Complete ---")
        print(f"Total Attempts: {total_attempts}")
        print(f"Valid Factors Generated: {valid_count}")
        print(f"Best Factor: {best_factor}")
        print(f"Best IR: {best_ir:.4f}")
        print(f"Best IC: {best_ic:.4f}")
