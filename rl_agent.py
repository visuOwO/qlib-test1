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
from factor_env import init_worker
from dqn_model import DQN, RNN_DQN, RNN_DQN_Combined
from qcm_module import QCMModule
from quantile_network import QuantileNetwork
from linear_model import fit_linear_ic, evaluate_factor_quality

import traceback

class DeepQLearningAgent:
    def __init__(self, env, hidden_dim=128, lr=1e-3, gamma=0.99, epsilon=1.0, top_k=10):
        self.env = env
        self.builder = FactorBuilder(max_seq_len=10, features=env.raw_features)

        self.action_dim = len(self.builder.action_map)

        self.device = torch.device("cuda" if torch.cuda.is_available() else ("xpu" if torch.xpu.is_available() else "cpu"))
        print(f"Agent using device: {self.device}")

        # 使用组合模型：RNN 处理因子序列 + DQN 选择动作（单网络模式）
        self.policy_net = RNN_DQN_Combined(
            action_dim=self.action_dim, 
            rnn_hidden_dim=hidden_dim, 
            dqn_hidden_dim=hidden_dim
        ).to(self.device)

        # ========== QCM 模块：用于估计奖励分布的方差 ==========
        # 分位数网络
        self.quantile_net = QuantileNetwork(
            action_dim=self.action_dim,
            embedding_dim=64,
            hidden_dim=hidden_dim
        ).to(self.device)
        
        # QCM 模块
        self.qcm = QCMModule(
            quantile_network=self.quantile_net,
            num_quantiles=32,
            tau_min=0.05,
            tau_max=0.95
        )
        
        # UCB 探索参数 λ
        self.ucb_lambda = 1.0

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = collections.deque(maxlen=2000)
        self.validator = FactorValidator(max_feature_repeat=3)

        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.batch_size = 64

        # Top K 因子设置
        self.top_k = top_k
        self.top_factors = []  # 存储 dict: expr, ir, ic, icir, mono, weight

        self.ic_scale = 1000    # IC放大倍数

    def select_action(self, state, valid_actions):
        if random.random() < self.epsilon:
            return random.choice(valid_actions)
        
        with torch.no_grad():
            state_t = torch.LongTensor(state).unsqueeze(0).to(self.device)
            
            # 1. 获取 Q 值（来自 DQN 网络）
            q_values = self.policy_net(state_t)  # [1, action_dim]
            
            # 2. 获取方差估计（来自 QCM 模块）
            variance = self.qcm(state_t)  # [1, action_dim]
            
            # 3. 计算 UCB 风格的分数：Score = Q + λ * sqrt(Variance)
            # 使用 clamp 避免负方差导致的 NaN
            std = torch.sqrt(torch.clamp(variance, min=1e-8))
            scores = q_values + self.ucb_lambda * std
            
            # 4. 屏蔽无效动作
            full_mask = torch.full(scores.shape, -float('inf')).to(self.device)
            full_mask[0, valid_actions] = 0  # Unmask valid
            
            masked_scores = scores + full_mask
            return masked_scores.argmax().item()
    
    def set_ucb_lambda(self, value):
        """设置 UCB 探索参数 λ"""
        self.ucb_lambda = value
        print(f"UCB lambda updated to: {value}")

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

        # Compute V(s_{t+1}) for all next states using policy_net (单网络模式)
        with torch.no_grad():
            # 直接用 policy_net 计算下一状态的最大 Q 值
            next_state_values = self.policy_net(next_state_batch).max(1)[0].unsqueeze(1)
        
        # Compute expected Q
        expected_q_values = reward_batch + (self.gamma * next_state_values * (1 - done_batch))

        loss = F.mse_loss(q_values, expected_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def _get_top_factor_exprs(self):
        return [factor["expr"] for factor in self.top_factors]

    def _sync_weights(self, weights_by_expr):
        for factor in self.top_factors:
            factor["weight"] = weights_by_expr.get(factor["expr"], 0.0)

    def _update_top_k_with_linear_model(self, new_factor):
        """
        使用 Top-K + 新因子做线性拟合，按权重绝对值决定是否替换。
        返回 (reward, accepted, removed_expr, old_ic, new_ic)
        """
        expr = new_factor["expr"]
        if any(factor["expr"] == expr for factor in self.top_factors):
            return 0.0, False, None, 0.0, 0.0

        old_exprs = self._get_top_factor_exprs()
        old_ic = 0.0
        old_weights = {}
        if old_exprs:
            old_ic, old_weights = fit_linear_ic(
                old_exprs,
                self.env.start_date,
                self.env.end_date,
                provider_uri=self.env.provider_uri,
                csi500_membership_path=self.env.csi500_membership_path,
            )
            if old_ic is None:
                return -2.0, False, None, 0.0, 0.0

        if len(old_exprs) < self.top_k:
            new_exprs = old_exprs + [expr]
            new_ic, new_weights = fit_linear_ic(
                new_exprs,
                self.env.start_date,
                self.env.end_date,
                provider_uri=self.env.provider_uri,
                csi500_membership_path=self.env.csi500_membership_path,
            )
            if new_ic is None:
                return -2.0, False, None, old_ic, old_ic
            self.top_factors.append(new_factor)
            self._sync_weights(new_weights)
            return new_ic - old_ic, True, None, old_ic, new_ic

        candidate_exprs = old_exprs + [expr]
        candidate_ic, candidate_weights = fit_linear_ic(
            candidate_exprs,
            self.env.start_date,
            self.env.end_date,
            provider_uri=self.env.provider_uri,
            csi500_membership_path=self.env.csi500_membership_path,
        )
        if candidate_ic is None:
            self._sync_weights(old_weights)
            return -2.0, False, None, old_ic, old_ic

        new_weight = abs(candidate_weights.get(expr, 0.0))
        min_expr = min(old_exprs, key=lambda e: abs(candidate_weights.get(e, 0.0)))
        min_weight = abs(candidate_weights.get(min_expr, 0.0))

        if new_weight < min_weight:
            self._sync_weights(old_weights)
            return 0.0, False, None, old_ic, old_ic

        updated_exprs = [e for e in old_exprs if e != min_expr] + [expr]
        new_ic, new_weights = fit_linear_ic(
            updated_exprs,
            self.env.start_date,
            self.env.end_date,
            provider_uri=self.env.provider_uri,
            csi500_membership_path=self.env.csi500_membership_path,
        )
        if new_ic is None:
            self._sync_weights(old_weights)
            return -2.0, False, None, old_ic, old_ic

        self.top_factors = [factor for factor in self.top_factors if factor["expr"] != min_expr]
        self.top_factors.append(new_factor)
        self._sync_weights(new_weights)
        return new_ic - old_ic, True, min_expr, old_ic, new_ic

    def _save_top_factors(self, filepath="factors.txt"):
        """保存 top k 因子到文件"""
        if not self.top_factors:
            print("No factors to save.")
            return

        # 获取脚本所在目录的父目录（即项目根目录）
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_path = os.path.join(script_dir, filepath)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"# Top {len(self.top_factors)} Factors (linear model weight-based)\n")
            f.write(f"# Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("# Format: Weight | IR | IC | ICIR | Monotonicity | Expression\n")
            f.write("=" * 80 + "\n\n")

            for i, factor in enumerate(self.top_factors, 1):
                f.write(
                    f"{i}. Weight={factor.get('weight', 0.0):.6f} | IR={factor['ir']:.4f} | "
                    f"IC={factor['ic']:.4f} | ICIR={factor['icir']:.4f} | Mono={factor['mono']:.4f}\n"
                )
                f.write(f"   {factor['expr']}\n\n")

        print(f"Top {len(self.top_factors)} factors saved to: {output_path}")

    def train(self, target_valid_episodes=50, max_attempts=1000, num_workers=None):
        if num_workers is None:
            num_workers = os.cpu_count()

        print(f"\n--- Starting Parallel Deep Q-Learning (Target: {target_valid_episodes} Valid Factors, Workers: {num_workers}) ---")

        # 重置 top k 因子列表
        self.top_factors = []

        valid_count = 0
        total_attempts = 0

        seen_factors = set()
        pending_futures = {}  # future -> (state, action, next_state, done, expr)

        with concurrent.futures.ProcessPoolExecutor(
            max_workers=num_workers,
            initializer=init_worker,  # 指定初始化函数
            initargs=(
                self.env.provider_uri,
                self.env.start_date,
                self.env.end_date,
                self.env.benchmark_code,
                self.env.csi500_membership_path,
            ) # 传递参数
        ) as executor:
            
            while (valid_count < target_valid_episodes and total_attempts < max_attempts) or pending_futures:
                did_work = False
                
                # --- 1. Process Completed Futures ---
                # Check for completed futures without blocking
                completed_futures = [f for f in pending_futures if f.done()]
                if completed_futures:
                    did_work = True # 有任务完成
                
                for future in completed_futures:
                    state, action, next_state, done, expr = pending_futures.pop(future)
                    
                    try:
                        price_corr, monotonicity = future.result()
                        if price_corr is None or monotonicity is None:
                            raise ValueError("Factor quality evaluation failed.")

                        is_correlated = abs(price_corr) > 0.9

                        # === [新增] 单调性检查 ===
                        is_monotonic = monotonicity > 0.0

                        if is_correlated:
                            reward = -5.0
                        elif not is_monotonic:
                            reward = -1.0
                        else:
                            new_factor = {
                                "expr": expr,
                                "ir": 0.0,
                                "ic": 0.0,
                                "icir": 0.0,
                                "mono": monotonicity,
                                "weight": 0.0,
                            }
                            delta_ic, accepted, removed_expr, old_ic, new_ic = self._update_top_k_with_linear_model(new_factor)

                            if accepted:
                                valid_count += 1

                                base_reward = 1.0  # 只要被接受，保底给 1 分

                                scaled_gain = delta_ic * self.ic_scale
                                reward = base_reward + scaled_gain

                                if removed_expr:
                                    print(
                                        f"Top-K Replaced: {removed_expr} -> {expr} | "
                                        f"Delta IC: {reward:.4f} (old IC={old_ic:.4f}, new IC={new_ic:.4f})"
                                    )
                                else:
                                    print(
                                        f"Top-K Added: {expr} | "
                                        f"Delta IC: {reward:.4f} (old IC={old_ic:.4f}, new IC={new_ic:.4f})"
                                    )
                            else:
                                print(
                                    f"Top-K Unchanged: {expr} | "
                                    f"Delta IC: {reward:.4f} (old IC={old_ic:.4f}, new IC={new_ic:.4f})"
                                )
                            
                    except Exception as e:
                        print(f"Error evaluating factor {expr}: {e}")
                        reward = -2.0 # Penalty for causing an error
                        traceback.print_exc()

                    self.memory.append((state, action, reward, next_state, done))
                    print("Optimizing model......")
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
                                print(f"Total attempts: {total_attempts}, Invalid factor generated: {expr}.")
                                self.optimize_model()
                            elif expr in seen_factors:
                                print(f"Total attempts: {total_attempts}, Duplicate factor generated: {expr}. Skipping update.")
                                pass
                            else:
                                # This is a new, valid factor to be evaluated
                                seen_factors.add(expr)
                                
                                # Epsilon Decay: 仅在生成有效因子时衰减
                                self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
                                
                                print(f"Attempt {total_attempts} (Submitting for Eval): {expr}")
                                future = executor.submit(
                                    evaluate_factor_quality,
                                    expr, 
                                    self.env.start_date, 
                                    self.env.end_date,
                                    self.env.provider_uri,
                                    self.env.csi500_membership_path,
                                )
                                pending_futures[future] = (prev_state, action, next_temp_state, done, expr)
                        else:
                            # 中间步骤也写入经验池，以便学习构建序列
                            self.memory.append((prev_state, action, 0.0, next_temp_state, False))
                            self.optimize_model()
                        
                        temp_state = next_temp_state

                if total_attempts % 10 == 0 and total_attempts > 0 and did_work:
                     print(f"Progress: {valid_count}/{target_valid_episodes} Valid | Attempts: {total_attempts} | Pending: {len(pending_futures)} | Epsilon: {self.epsilon:.2f}")

                if not did_work:
                    pass

        print("\n--- Training Complete ---")
        print(f"Total Attempts: {total_attempts}")
        print(f"Valid Factors Generated: {valid_count}")

        # 保存 top k 因子到文件
        self._save_top_factors("factors.txt")

        # 打印 top k 因子摘要
        if self.top_factors:
            print(f"\n=== Top {len(self.top_factors)} Factors ===")
            for i, factor in enumerate(self.top_factors, 1):
                print(
                    f"{i}. Weight={factor.get('weight', 0.0):.6f} | IR={factor['ir']:.4f} | "
                    f"IC={factor['ic']:.4f} | ICIR={factor['icir']:.4f} | Mono={factor['mono']:.4f}"
                )
                print(f"   {factor['expr']}")

        return {
            "top_factors": [
                (f["expr"], f["ir"], f["ic"], f["icir"], f["mono"], f.get("weight", 0.0))
                for f in self.top_factors
            ],  # (expr, ir, ic, icir, mono, weight)
            "total_attempts": total_attempts,
            "valid_count": valid_count,
        }
