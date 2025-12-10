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