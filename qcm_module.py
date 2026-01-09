import torch
import torch.nn as nn
import math
from scipy import stats
import numpy as np

from quantile_network import QuantileNetwork


class QCMModule(nn.Module):
    """
    QCM (Quantile-based Cornish-Fisher Model) 模块
    
    工作流程：
    1. 前向传播：将状态 x_t 输入 QuantileNetwork，生成 K 个 τ 对应的分位值
    2. QCM 计算：使用 Cornish-Fisher 公式，通过线性回归求解方差
    3. 返回：估计的方差 ĥ
    
    Cornish-Fisher 展开式：
    ξ_τ ≈ μ + σ · z_τ + γ₁ · (z_τ² - 1) / 6 + ...
    
    简化形式下，斜率近似于标准差 σ
    """
    
    def __init__(
        self, 
        quantile_network: QuantileNetwork,
        num_quantiles: int = 32,
        tau_min: float = 0.05,
        tau_max: float = 0.95
    ):
        """
        初始化 QCM 模块
        
        Args:
            quantile_network: 预训练的分位数网络
            num_quantiles: 采样的分位数数量 K
            tau_min: τ 的最小值
            tau_max: τ 的最大值
        """
        super(QCMModule, self).__init__()
        
        self.quantile_network = quantile_network
        self.num_quantiles = num_quantiles
        self.tau_min = tau_min
        self.tau_max = tau_max
        
        # 预计算固定的 τ 值（均匀分布）
        # 例如：0.05, 0.08, 0.11, ..., 0.95
        taus = torch.linspace(tau_min, tau_max, num_quantiles)
        self.register_buffer('fixed_taus', taus)
        
        # 预计算对应的标准正态分位数 z_τ = Φ⁻¹(τ)
        z_taus = torch.tensor([stats.norm.ppf(t.item()) for t in taus], dtype=torch.float32)
        self.register_buffer('z_taus', z_taus)
        
    def forward(self, x, action_idx=None):
        """
        前向传播：估计奖励分布的方差
        
        Args:
            x: 状态序列 (Token序列), shape: [batch_size, seq_len]
            action_idx: 可选，指定动作索引。如果为None，返回所有动作的方差
                        shape: [batch_size] 或 int
            
        Returns:
            variance_hat: 估计的方差 ĥ
                如果 action_idx 为 None: shape [batch_size, action_dim]
                否则: shape [batch_size]
        """
        batch_size = x.size(0)
        device = x.device
        
        # ========== 1. 获取分位数预测 ==========
        # 扩展 τ 到 batch 维度
        # fixed_taus: [K] -> [batch, K]
        taus = self.fixed_taus.unsqueeze(0).expand(batch_size, -1)
        
        # 通过分位数网络获取预测值
        # quantile_values: [batch, K, action_dim]
        quantile_values = self.quantile_network(x, taus)
        
        # ========== 2. 应用 Cornish-Fisher 公式计算方差 ==========
        if action_idx is not None:
            # 只计算指定动作的方差
            if isinstance(action_idx, int):
                # 单个动作索引，应用于所有 batch
                quantile_values = quantile_values[:, :, action_idx]  # [batch, K]
            else:
                # 每个 batch 有不同的动作索引
                # action_idx: [batch]
                batch_indices = torch.arange(batch_size, device=device)
                quantile_values = quantile_values[batch_indices, :, action_idx]  # [batch, K]
            
            variance_hat = self._compute_variance_batch(quantile_values)  # [batch]
        else:
            # 计算所有动作的方差
            action_dim = quantile_values.size(2)
            variance_hat = torch.zeros(batch_size, action_dim, device=device)
            
            for a in range(action_dim):
                qv = quantile_values[:, :, a]  # [batch, K]
                variance_hat[:, a] = self._compute_variance_batch(qv)
        
        return variance_hat
    
    def _compute_variance_batch(self, quantile_values):
        """
        使用 Cornish-Fisher 公式计算方差（批量版本）
        
        Cornish-Fisher 简化公式：
        ξ_τ ≈ μ + σ · z_τ
        
        这是一个简单的线性回归问题：
        ξ_τ = a + b · z_τ
        
        其中 b = σ（标准差），方差 = σ²
        
        使用最小二乘法：
        b = Σ(z_τ - z̄)(ξ_τ - ξ̄) / Σ(z_τ - z̄)²
        
        Args:
            quantile_values: [batch_size, K]
            
        Returns:
            variance: [batch_size]
        """
        batch_size = quantile_values.size(0)
        device = quantile_values.device
        
        # z_taus: [K] -> [1, K]
        z = self.z_taus.unsqueeze(0).to(device)  # [1, K]
        
        # 中心化
        z_mean = z.mean(dim=1, keepdim=True)  # [1, 1]
        q_mean = quantile_values.mean(dim=1, keepdim=True)  # [batch, 1]
        
        z_centered = z - z_mean  # [1, K]
        q_centered = quantile_values - q_mean  # [batch, K]
        
        # 计算斜率 b = Cov(z, ξ) / Var(z)
        # 分子：Σ(z - z̄)(ξ - ξ̄)
        numerator = (z_centered * q_centered).sum(dim=1)  # [batch]
        
        # 分母：Σ(z - z̄)²
        denominator = (z_centered ** 2).sum(dim=1)  # [1] -> 广播到 [batch]
        
        # 斜率 = 标准差 σ
        sigma = numerator / (denominator + 1e-8)  # [batch]
        
        # 方差 = σ²
        variance = sigma ** 2
        
        return variance
    
    def get_distribution_stats(self, x, action_idx=None):
        """
        获取分布的完整统计信息
        
        Args:
            x: 状态序列
            action_idx: 可选，指定动作索引
            
        Returns:
            dict: 包含 mean, std, variance, skewness(近似) 的字典
        """
        batch_size = x.size(0)
        device = x.device
        
        # 获取分位数预测
        taus = self.fixed_taus.unsqueeze(0).expand(batch_size, -1)
        quantile_values = self.quantile_network(x, taus)
        
        if action_idx is not None:
            if isinstance(action_idx, int):
                quantile_values = quantile_values[:, :, action_idx]
            else:
                batch_indices = torch.arange(batch_size, device=device)
                quantile_values = quantile_values[batch_indices, :, action_idx]
        
        # 计算统计量
        mean = quantile_values.mean(dim=1)
        std = self._compute_std_batch(quantile_values)
        variance = std ** 2
        
        # 近似偏度（使用分位数的不对称性）
        median_idx = self.num_quantiles // 2
        q_low = quantile_values[:, :median_idx].mean(dim=1)
        q_high = quantile_values[:, median_idx:].mean(dim=1)
        skewness_approx = q_high + q_low - 2 * mean
        
        return {
            'mean': mean,
            'std': std,
            'variance': variance,
            'skewness_approx': skewness_approx,
            'quantile_values': quantile_values
        }
    
    def _compute_std_batch(self, quantile_values):
        """
        计算标准差（与 _compute_variance_batch 相同，但返回 σ 而不是 σ²）
        """
        device = quantile_values.device
        z = self.z_taus.unsqueeze(0).to(device)
        
        z_mean = z.mean(dim=1, keepdim=True)
        q_mean = quantile_values.mean(dim=1, keepdim=True)
        
        z_centered = z - z_mean
        q_centered = quantile_values - q_mean
        
        numerator = (z_centered * q_centered).sum(dim=1)
        denominator = (z_centered ** 2).sum(dim=1)
        
        sigma = numerator / (denominator + 1e-8)
        
        # 返回绝对值，确保标准差为正
        return torch.abs(sigma)


class QCMWithCornishFisherExpansion(QCMModule):
    """
    带完整 Cornish-Fisher 展开的 QCM 模块
    
    完整的 Cornish-Fisher 展开式考虑了高阶矩：
    ξ_τ ≈ μ + σ · [z_τ + γ₁(z_τ² - 1)/6 + γ₂(z_τ³ - 3z_τ)/24 - γ₁²(2z_τ³ - 5z_τ)/36]
    
    其中：
    - γ₁ = 偏度 (skewness)
    - γ₂ = 超额峰度 (excess kurtosis)
    
    通过多元线性回归同时估计 σ, γ₁, γ₂
    """
    
    def __init__(
        self, 
        quantile_network: QuantileNetwork,
        num_quantiles: int = 32,
        tau_min: float = 0.05,
        tau_max: float = 0.95,
        include_higher_moments: bool = True
    ):
        super().__init__(quantile_network, num_quantiles, tau_min, tau_max)
        
        self.include_higher_moments = include_higher_moments
        
        # 预计算回归设计矩阵的各项
        z = self.z_taus
        
        if include_higher_moments:
            # 设计矩阵列：[1, z, (z²-1)/6, (z³-3z)/24]
            ones = torch.ones_like(z)
            z1 = z
            z2 = (z ** 2 - 1) / 6  # skewness 项
            z3 = (z ** 3 - 3 * z) / 24  # kurtosis 项
            
            design_matrix = torch.stack([ones, z1, z2, z3], dim=1)  # [K, 4]
        else:
            # 简化版：只有 [1, z]
            ones = torch.ones_like(z)
            design_matrix = torch.stack([ones, z], dim=1)  # [K, 2]
        
        self.register_buffer('design_matrix', design_matrix)
        
        # 预计算 (X'X)^(-1) X' 用于最小二乘
        X = design_matrix
        XtX_inv = torch.inverse(X.T @ X)
        projection_matrix = XtX_inv @ X.T  # [num_params, K]
        self.register_buffer('projection_matrix', projection_matrix)
    
    def _compute_variance_batch(self, quantile_values):
        """
        使用完整 Cornish-Fisher 展开计算方差
        
        通过最小二乘法求解：
        β = (X'X)^(-1) X' ξ
        
        其中 β = [μ, σ, σ·γ₁, σ·γ₂]' (近似)
        """
        device = quantile_values.device
        
        # projection_matrix: [num_params, K]
        # quantile_values: [batch, K]
        proj = self.projection_matrix.to(device)
        
        # β = proj @ ξ.T -> [num_params, batch] -> [batch, num_params]
        beta = (proj @ quantile_values.T).T  # [batch, num_params]
        
        # β[1] 对应 σ（标准差）
        sigma = beta[:, 1]
        
        # 方差 = σ²
        variance = sigma ** 2
        
        return variance
    
    def get_full_moments(self, x, action_idx=None):
        """
        获取完整的矩估计（均值、方差、偏度、峰度）
        
        Returns:
            dict: 包含所有估计矩的字典
        """
        batch_size = x.size(0)
        device = x.device
        
        taus = self.fixed_taus.unsqueeze(0).expand(batch_size, -1)
        quantile_values = self.quantile_network(x, taus)
        
        if action_idx is not None:
            if isinstance(action_idx, int):
                quantile_values = quantile_values[:, :, action_idx]
            else:
                batch_indices = torch.arange(batch_size, device=device)
                quantile_values = quantile_values[batch_indices, :, action_idx]
        
        # 最小二乘求解
        proj = self.projection_matrix.to(device)
        beta = (proj @ quantile_values.T).T  # [batch, num_params]
        
        mu = beta[:, 0]  # 均值
        sigma = torch.abs(beta[:, 1])  # 标准差
        
        result = {
            'mean': mu,
            'std': sigma,
            'variance': sigma ** 2
        }
        
        if self.include_higher_moments and beta.size(1) >= 4:
            # 偏度和峰度需要从系数中提取
            # 注意：这里的系数是 σ·γ₁/6 和 σ·γ₂/24 的近似
            gamma1_coef = beta[:, 2]  # ≈ σ·γ₁/6
            gamma2_coef = beta[:, 3]  # ≈ σ·γ₂/24
            
            # 反推偏度和峰度
            skewness = 6 * gamma1_coef / (sigma + 1e-8)
            kurtosis = 24 * gamma2_coef / (sigma + 1e-8)
            
            result['skewness'] = skewness
            result['kurtosis'] = kurtosis
        
        return result


# ========== 使用示例 ==========
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 创建分位数网络
    action_dim = 50
    quantile_net = QuantileNetwork(
        action_dim=action_dim,
        embedding_dim=64,
        hidden_dim=128
    ).to(device)
    
    # 创建 QCM 模块
    qcm = QCMModule(
        quantile_network=quantile_net,
        num_quantiles=32,
        tau_min=0.05,
        tau_max=0.95
    ).to(device)
    
    # 模拟数据
    batch_size = 16
    seq_len = 10
    states = torch.randint(0, action_dim + 1, (batch_size, seq_len)).to(device)
    
    # 计算方差估计
    print("\n=== 基础 QCM 测试 ===")
    variance_all = qcm(states)
    print(f"所有动作的方差估计 shape: {variance_all.shape}")
    print(f"第一个样本前5个动作的方差: {variance_all[0, :5].detach().cpu().numpy()}")
    
    # 指定动作索引
    action_idx = torch.randint(0, action_dim, (batch_size,)).to(device)
    variance_specific = qcm(states, action_idx)
    print(f"指定动作的方差估计 shape: {variance_specific.shape}")
    
    # 获取完整分布统计
    stats_dict = qcm.get_distribution_stats(states, action_idx=0)
    print(f"\n动作 0 的分布统计:")
    print(f"  均值: {stats_dict['mean'][:3].detach().cpu().numpy()}")
    print(f"  标准差: {stats_dict['std'][:3].detach().cpu().numpy()}")
    print(f"  方差: {stats_dict['variance'][:3].detach().cpu().numpy()}")
    
    # 测试带高阶矩的 QCM
    print("\n=== Cornish-Fisher 展开 QCM 测试 ===")
    qcm_cf = QCMWithCornishFisherExpansion(
        quantile_network=quantile_net,
        num_quantiles=32,
        include_higher_moments=True
    ).to(device)
    
    moments = qcm_cf.get_full_moments(states, action_idx=0)
    print(f"完整矩估计:")
    print(f"  均值: {moments['mean'][:3].detach().cpu().numpy()}")
    print(f"  标准差: {moments['std'][:3].detach().cpu().numpy()}")
    print(f"  偏度: {moments.get('skewness', 'N/A')[:3].detach().cpu().numpy() if 'skewness' in moments else 'N/A'}")
    print(f"  峰度: {moments.get('kurtosis', 'N/A')[:3].detach().cpu().numpy() if 'kurtosis' in moments else 'N/A'}")
