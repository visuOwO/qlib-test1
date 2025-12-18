import torch
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
import collections
import concurrent.futures
import os
import time
from factor_builder import FactorBuilder
from factor_validator import FactorValidator
from factor_env import evaluate_factor_mp, init_worker
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
            # 1. 用 Policy Net 决定哪个动作最好 (argmax)
            next_actions = self.policy_net(next_state_batch).argmax(1).unsqueeze(1)
            
            # 2. 用 Target Net 计算这个动作的价值
            next_state_values = self.target_net(next_state_batch).gather(1, next_actions)
        
        # Compute expected Q
        expected_q_values = reward_batch + (self.gamma * next_state_values * (1 - done_batch))

        loss = F.mse_loss(q_values, expected_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self, target_valid_episodes=50, max_attempts=1000, num_workers=None):
        if num_workers is None:
            num_workers = os.cpu_count()
        
        print(f"\n--- Starting Parallel Deep Q-Learning (Target: {target_valid_episodes} Valid Factors, Workers: {num_workers}) ---")
        
        best_ir = -float('inf')
        best_ic = -float('inf')
        best_factor = ""
        best_icir = -float('inf')
        valid_count = 0
        total_attempts = 0
        
        seen_factors = set()
        pending_futures = {}  # future -> (state, action, next_state, done, expr)

        with concurrent.futures.ProcessPoolExecutor(
            max_workers=num_workers,
            initializer=init_worker,  # 指定初始化函数
            initargs=(self.env.provider_uri, self.env.start_date, self.env.end_date, self.env.benchmark_code) # 传递参数
        ) as executor:
            
            while valid_count < target_valid_episodes and total_attempts < max_attempts:
                did_work = False
                
                # --- 1. Process Completed Futures ---
                # Check for completed futures without blocking
                completed_futures = [f for f in pending_futures if f.done()]
                if completed_futures:
                    did_work = True # 有任务完成
                
                for future in completed_futures:
                    state, action, next_state, done, expr = pending_futures.pop(future)
                    
                    try:
                        ir, ic, icir, price_corr = future.result()
                        
                        is_correlated = abs(price_corr) > 0.6

                        if ir > best_ir and not is_correlated:
                            best_ir = ir
                            best_ic = ic
                            best_icir = icir
                            best_factor = expr
                            print(f"New Best! IR: {best_ir:.4f} | IC: {best_ic:.4f} | ICIR: {icir:.4f} | Corr: {price_corr:.4f} | {expr}")

                        if is_correlated:
                            reward = -5.0
                        elif ir > 0:
                            print(f"Valid & New: {expr} | IC: {ic:.4f} | ICIR: {icir:.4f}")
                            valid_count += 1
                            reward = ir * 20
                        else:
                            reward = -1.0
                            
                    except Exception as e:
                        print(f"Error evaluating factor {expr}: {e}")
                        reward = -2.0 # Penalty for causing an error

                    self.memory.append((state, action, reward, next_state, done))
                    self.optimize_model()

                # --- 2. Generate New Factors if Pool has Capacity ---
                while len(pending_futures) < num_workers and total_attempts < max_attempts:
                    did_work = True # 提交了新任务
                    state = self.builder.reset()
                    total_attempts += 1
                    
                    # Generate one full expression
                    done = False
                    temp_state = state
                    while not done:
                        valid_actions = self.builder.get_valid_actions()
                        prev_state = temp_state
                        action = self.select_action(temp_state, valid_actions)
                        next_temp_state, done = self.builder.step(action)
                        
                        if done:
                            expr = self.builder.build_expression()
                            
                            # --- Pre-validation checks ---
                            if not self.validator.validate(expr):
                                reward = -10.0
                                self.memory.append((prev_state, action, reward, next_temp_state, done))
                                self.optimize_model()
                            elif expr in seen_factors:
                                reward = -5.0
                                self.memory.append((prev_state, action, reward, next_temp_state, done))
                                self.optimize_model()
                            else:
                                # This is a new, valid factor to be evaluated
                                seen_factors.add(expr)
                                print(f"Attempt {total_attempts} (Submitting for Eval): {expr}")
                                future = executor.submit(
                                    evaluate_factor_mp, 
                                    expr, 
                                    self.env.start_date, 
                                    self.env.end_date
                                )
                                pending_futures[future] = (prev_state, action, next_temp_state, done, expr)
                        else:
                            # 中间步骤也写入经验池，以便学习构建序列
                            self.memory.append((prev_state, action, 0.0, next_temp_state, False))
                            self.optimize_model()
                        
                        temp_state = next_temp_state

                # Epsilon Decay
                self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
                
                # Target Network Update & Progress Print
                if total_attempts % 10 == 0 and total_attempts > 0:
                    self.target_net.load_state_dict(self.policy_net.state_dict())
                    print(f"Progress: {valid_count}/{target_valid_episodes} Valid | Attempts: {total_attempts} | Pending: {len(pending_futures)} | Epsilon: {self.epsilon:.2f}")

                if not did_work:
                    time.sleep(0.1)

        print("\n--- Training Complete ---")
        print(f"Total Attempts: {total_attempts}")
        print(f"Valid Factors Generated: {valid_count}")
        print(f"Best Factor: {best_factor}")
        print(f"Best IR: {best_ir:.4f}")
        print(f"Best IC: {best_ic:.4f}")
        print(f"Best ICIR: {best_icir:.4f}")

        return {
            "best_factor": best_factor,
            "best_ir": best_ir,
            "best_ic": best_ic,
            "best_icir": best_icir,
            "total_attempts": total_attempts,
            "valid_count": valid_count,
        }
