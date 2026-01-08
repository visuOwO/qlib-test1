import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    
class RNN_DQN(nn.Module):
    def __init__(self, action_dim, embedding_dim=64, hidden_dim=128):
        super(RNN_DQN, self).__init__()
        # 1. Embedding: 将动作ID (整数) 映射为密集向量
        # +1 是为了处理 padding (0) 或者特殊的起始符
        self.embedding = nn.Embedding(num_embeddings=action_dim + 1, embedding_dim=embedding_dim)
        
        # 2. GRU: 处理序列信息
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        
        # 3. Head: 输出 Q 值
        self.fc = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        # x shape: [batch_size, seq_len] (整数索引)
        embeds = self.embedding(x)  # -> [batch, seq, embed_dim]
        
        # 只需要取 GRU 最后一个时间步的输出
        _, h_n = self.gru(embeds)   # h_n: [1, batch, hidden]
        
        # 去掉第一维
        h_n = h_n.squeeze(0)        # -> [batch, hidden]
        
        return self.fc(h_n)


class RNN_DQN_Combined(nn.Module):
    """
    组合模型：先用 RNN 处理因子序列（提取时序特征），
    然后将输出传递给 DQN 网络选择最佳动作。
    
    流程：
    1. Embedding: 将动作/因子 ID 映射为向量
    2. GRU: 处理序列信息，提取时序特征
    3. DQN: 基于 RNN 隐藏状态选择最佳动作
    4. 根据选中的动作生成新因子
    """
    def __init__(self, action_dim, embedding_dim=64, rnn_hidden_dim=128, dqn_hidden_dim=128):
        super(RNN_DQN_Combined, self).__init__()
        
        self.action_dim = action_dim
        
        # ========== RNN 部分：处理因子序列 ==========
        # Embedding: 将动作/因子 ID 映射为密集向量
        self.embedding = nn.Embedding(num_embeddings=action_dim + 1, embedding_dim=embedding_dim)
        
        # GRU: 处理序列信息，提取时序特征
        self.gru = nn.GRU(embedding_dim, rnn_hidden_dim, batch_first=True)
        
        # ========== DQN 部分：基于 RNN 输出选择动作 ==========
        # 将 RNN 的隐藏状态作为 DQN 的输入
        self.dqn_fc1 = nn.Linear(rnn_hidden_dim, dqn_hidden_dim)
        self.dqn_fc2 = nn.Linear(dqn_hidden_dim, dqn_hidden_dim)
        self.dqn_output = nn.Linear(dqn_hidden_dim, action_dim)
        
    def forward(self, x, return_rnn_features=False):
        """
        前向传播
        
        Args:
            x: 因子序列, shape: [batch_size, seq_len] (整数索引)
            return_rnn_features: 是否同时返回 RNN 的隐藏特征
            
        Returns:
            q_values: 每个动作的 Q 值, shape: [batch_size, action_dim]
            rnn_features (可选): RNN 隐藏状态, shape: [batch_size, rnn_hidden_dim]
        """
        # ========== 第一阶段：RNN 处理因子序列 ==========
        # Embedding
        embeds = self.embedding(x)  # -> [batch, seq, embed_dim]
        
        # GRU 提取时序特征，取最后一个时间步的隐藏状态
        _, h_n = self.gru(embeds)   # h_n: [1, batch, rnn_hidden]
        rnn_features = h_n.squeeze(0)  # -> [batch, rnn_hidden]
        
        # ========== 第二阶段：DQN 选择最佳动作 ==========
        dqn_out = F.relu(self.dqn_fc1(rnn_features))
        dqn_out = F.relu(self.dqn_fc2(dqn_out))
        q_values = self.dqn_output(dqn_out)  # -> [batch, action_dim]
        
        if return_rnn_features:
            return q_values, rnn_features
        return q_values
    
    def select_action(self, x, epsilon=0.0):
        """
        选择动作 (带 epsilon-greedy 策略)
        
        Args:
            x: 因子序列, shape: [batch_size, seq_len]
            epsilon: 探索概率
            
        Returns:
            actions: 选中的动作索引, shape: [batch_size]
        """
        if torch.rand(1).item() < epsilon:
            # 随机探索
            batch_size = x.size(0)
            return torch.randint(0, self.action_dim, (batch_size,))
        else:
            # 贪婪选择
            with torch.no_grad():
                q_values = self.forward(x)
                return q_values.argmax(dim=1)
    
    def get_next_factor(self, x, epsilon=0.0):
        """
        根据当前因子序列，选择动作并生成新的因子
        
        Args:
            x: 当前因子序列, shape: [batch_size, seq_len]
            epsilon: 探索概率
            
        Returns:
            new_factor_seq: 包含新因子的序列, shape: [batch_size, seq_len + 1]
            action: 选中的动作 (即新因子), shape: [batch_size]
        """
        action = self.select_action(x, epsilon)
        
        # 将新动作追加到序列末尾
        # 注意：动作索引需要 +1 以匹配 embedding（0 用于 padding）
        new_factor = action.unsqueeze(1) + 1  # -> [batch, 1]
        new_factor_seq = torch.cat([x, new_factor], dim=1)
        
        return new_factor_seq, action