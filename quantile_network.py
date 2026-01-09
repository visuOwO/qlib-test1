import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class QuantileNetwork(nn.Module):
    """
    分位数网络 (Quantile Network)
    
    用于预测给定状态和分位数 τ 下的分位值。
    
    结构：
    1. 输入1: State x_t（Token序列）-> LSTM提取特征
    2. 输入2: 随机数 τ (0到1之间) -> Cosine Embedding编码
    3. 融合: state_feat * tau_feat (元素级别点乘)
    4. 输出: 预测的分位值
    
    使用 Quantile Huber Loss 进行优化。
    """
    
    def __init__(
        self, 
        action_dim, 
        embedding_dim=64, 
        hidden_dim=128, 
        tau_embed_dim=64,
        n_cos_features=64
    ):
        """
        初始化分位数网络
        
        Args:
            action_dim: 动作空间维度（Token的数量）
            embedding_dim: Token嵌入维度
            hidden_dim: LSTM隐藏层维度
            tau_embed_dim: τ嵌入维度（需要等于hidden_dim以便点乘）
            n_cos_features: Cosine特征数量
        """
        super(QuantileNetwork, self).__init__()
        
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.n_cos_features = n_cos_features
        
        # ========== 状态编码器（眼睛）：LSTM处理Token序列 ==========
        # Embedding: 将动作/Token ID 映射为密集向量
        self.embedding = nn.Embedding(
            num_embeddings=action_dim + 1, 
            embedding_dim=embedding_dim
        )
        
        # LSTM: 处理序列信息，提取时序特征
        self.lstm = nn.LSTM(
            input_size=embedding_dim, 
            hidden_size=hidden_dim, 
            batch_first=True
        )
        
        # ========== 分位数编码器（概率感知）：Cosine Embedding ==========
        # 使用 Cosine 函数将 τ 编码为向量
        # τ -> cos(i * π * τ) for i in [1, n_cos_features]
        # 然后通过线性层映射到与state_feat相同的维度
        self.tau_fc = nn.Linear(n_cos_features, hidden_dim)
        
        # ========== 输出头（嘴巴）：MLP预测分位值 ==========
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, x, tau):
        """
        前向传播
        
        Args:
            x: Token序列, shape: [batch_size, seq_len] (整数索引)
            tau: 分位数, shape: [batch_size, n_tau] 或 [batch_size, 1]
                 取值范围在 [0, 1] 之间
            
        Returns:
            quantile_values: 每个动作在给定τ下的分位值
                shape: [batch_size, n_tau, action_dim]
        """
        batch_size = x.size(0)
        n_tau = tau.size(1)
        
        # ========== 1. 状态特征提取（LSTM） ==========
        # Embedding
        embeds = self.embedding(x)  # -> [batch, seq, embed_dim]
        
        # LSTM提取时序特征
        _, (h_n, _) = self.lstm(embeds)  # h_n: [1, batch, hidden]
        state_feat = h_n.squeeze(0)  # -> [batch, hidden_dim]
        
        # ========== 2. τ特征编码（Cosine Embedding） ==========
        tau_feat = self._encode_tau(tau)  # -> [batch, n_tau, hidden_dim]
        
        # ========== 3. 特征融合（点乘） ==========
        # 扩展 state_feat 以匹配 tau 的数量
        # state_feat: [batch, hidden] -> [batch, n_tau, hidden]
        state_feat = state_feat.unsqueeze(1).expand(-1, n_tau, -1)
        
        # 元素级别点乘：让网络理解"在状态x_t下，概率为τ时"
        mixed_feat = state_feat * tau_feat  # -> [batch, n_tau, hidden_dim]
        
        # ========== 4. 输出分位值（MLP） ==========
        out = F.relu(self.fc1(mixed_feat))
        quantile_values = self.fc2(out)  # -> [batch, n_tau, action_dim]
        
        return quantile_values
    
    def _encode_tau(self, tau):
        """
        使用 Cosine Embedding 编码 τ
        
        τ -> cos(i * π * τ) for i in [1, n_cos_features]
        然后通过线性层映射
        
        Args:
            tau: shape [batch_size, n_tau], 取值范围 [0, 1]
            
        Returns:
            tau_feat: shape [batch_size, n_tau, hidden_dim]
        """
        batch_size = tau.size(0)
        n_tau = tau.size(1)
        
        # 生成 i = 1, 2, ..., n_cos_features
        i_pi = torch.arange(1, self.n_cos_features + 1, device=tau.device).float() * math.pi
        # i_pi: [n_cos_features]
        
        # 计算 cos(i * π * τ)
        # tau: [batch, n_tau] -> [batch, n_tau, 1]
        # i_pi: [n_cos_features] -> [1, 1, n_cos_features]
        tau_expanded = tau.unsqueeze(-1)  # [batch, n_tau, 1]
        cos_features = torch.cos(tau_expanded * i_pi)  # [batch, n_tau, n_cos_features]
        
        # 通过线性层映射到 hidden_dim
        tau_feat = F.relu(self.tau_fc(cos_features))  # [batch, n_tau, hidden_dim]
        
        return tau_feat
    
    def get_quantile_values(self, x, num_tau_samples=32):
        """
        采样多个 τ 并计算对应的分位值
        
        Args:
            x: Token序列, shape: [batch_size, seq_len]
            num_tau_samples: 采样的 τ 数量
            
        Returns:
            quantile_values: [batch_size, num_tau_samples, action_dim]
            tau: [batch_size, num_tau_samples]
        """
        batch_size = x.size(0)
        device = x.device
        
        # 随机采样 τ ∈ [0, 1]
        tau = torch.rand(batch_size, num_tau_samples, device=device)
        
        quantile_values = self.forward(x, tau)
        
        return quantile_values, tau
    

def quantile_huber_loss(predictions, targets, taus, kappa=1.0):
    """
    Quantile Huber Loss (分位数Huber损失)
    
    用于分位数回归的损失函数，结合了分位数损失和Huber损失的优点：
    - 分位数损失：根据预测误差的方向和τ进行加权
    - Huber损失：对异常值更加鲁棒
    
    公式：
    ρ_τ(u) = |τ - I(u < 0)| * L_κ(u)
    其中 L_κ(u) 是 Huber 损失：
        L_κ(u) = 0.5 * u² / κ,         if |u| <= κ
        L_κ(u) = |u| - 0.5 * κ,        otherwise
    
    Args:
        predictions: 预测的分位值, shape: [batch_size, n_tau, action_dim] 或 [batch_size, n_tau]
        targets: 目标值, shape: [batch_size, action_dim] 或 [batch_size]
        taus: 分位数, shape: [batch_size, n_tau]
        kappa: Huber损失的阈值参数，默认为1.0
        
    Returns:
        loss: 标量，Quantile Huber Loss
    """
    # 确保 targets 有正确的维度用于广播
    if targets.dim() == 1:
        # [batch] -> [batch, 1, 1]
        targets = targets.unsqueeze(1).unsqueeze(2)
    elif targets.dim() == 2:
        # [batch, action_dim] -> [batch, 1, action_dim]
        targets = targets.unsqueeze(1)
    
    # 计算预测误差 (TD error)
    # predictions: [batch, n_tau, action_dim]
    # targets: [batch, 1, action_dim]
    u = targets - predictions  # [batch, n_tau, action_dim]
    
    # 计算 Huber Loss
    abs_u = torch.abs(u)
    huber_loss = torch.where(
        abs_u <= kappa,
        0.5 * u ** 2 / kappa,
        abs_u - 0.5 * kappa
    )
    
    # 分位数权重：|τ - I(u < 0)|
    # taus: [batch, n_tau] -> [batch, n_tau, 1]
    taus = taus.unsqueeze(-1)
    
    # I(u < 0)：指示函数，u < 0 时为 1，否则为 0
    indicator = (u < 0).float()
    
    # 分位数加权
    quantile_weight = torch.abs(taus - indicator)
    
    # 最终损失
    loss = quantile_weight * huber_loss
    
    # 对所有维度求平均
    return loss.mean()


class QuantileNetworkTrainer:
    """
    分位数网络训练器
    
    封装训练逻辑，包括损失计算和优化步骤。
    """
    
    def __init__(self, network, learning_rate=1e-4, kappa=1.0):
        """
        初始化训练器
        
        Args:
            network: QuantileNetwork实例
            learning_rate: 学习率
            kappa: Quantile Huber Loss的阈值参数
        """
        self.network = network
        self.kappa = kappa
        self.optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
        
    def train_step(self, states, targets, num_tau_samples=32):
        """
        执行一步训练
        
        Args:
            states: 状态序列, shape: [batch_size, seq_len]
            targets: 目标值, shape: [batch_size] 或 [batch_size, action_dim]
            num_tau_samples: 采样的τ数量
            
        Returns:
            loss: 本次训练的损失值
        """
        self.network.train()
        self.optimizer.zero_grad()
        
        # 前向传播：采样τ并计算分位值
        quantile_values, taus = self.network.get_quantile_values(
            states, num_tau_samples
        )
        
        # 计算 Quantile Huber Loss
        loss = quantile_huber_loss(quantile_values, targets, taus, self.kappa)
        
        # 反向传播
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def evaluate(self, states, num_tau_samples=32):
        """
        评估模式：获取期望Q值
        
        Args:
            states: 状态序列
            num_tau_samples: 采样的τ数量
            
        Returns:
            expected_q: 期望Q值（对τ取平均）
        """
        self.network.eval()
        with torch.no_grad():
            quantile_values, _ = self.network.get_quantile_values(
                states, num_tau_samples
            )
            # 对τ维度取平均，得到期望Q值
            expected_q = quantile_values.mean(dim=1)
        return expected_q


# ========== 使用示例 ==========
if __name__ == "__main__":
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 创建网络
    action_dim = 50  # 假设有50个可能的token/动作
    network = QuantileNetwork(
        action_dim=action_dim,
        embedding_dim=64,
        hidden_dim=128,
        tau_embed_dim=128,
        n_cos_features=64
    ).to(device)
    
    # 创建训练器
    trainer = QuantileNetworkTrainer(network, learning_rate=1e-4)
    
    # 模拟数据
    batch_size = 32
    seq_len = 10
    
    # 随机生成状态序列（token索引）
    states = torch.randint(0, action_dim + 1, (batch_size, seq_len)).to(device)
    
    # 随机生成目标值
    targets = torch.randn(batch_size, action_dim).to(device)
    
    # 训练一步
    loss = trainer.train_step(states, targets, num_tau_samples=32)
    print(f"Training loss: {loss:.4f}")
    
    # 评估
    expected_q = trainer.evaluate(states)
    print(f"Expected Q-values shape: {expected_q.shape}")
    print(f"Expected Q-values (first sample): {expected_q[0][:5]}")  # 显示前5个动作的Q值
