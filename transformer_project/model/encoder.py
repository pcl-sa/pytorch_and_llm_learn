# 编码器层
#每一个 Encoder 层都包含2 个核心子层 + 2 组残差连接 + 层归一化
#子层 1：Multi-Head Self-Attention（多头自注意力）
#让序列中每个 token 都能关注到整个输入序列的所有 token，把孤立的词向量升级为「带全句上下文的向量」。
#Encoder 自注意力是无掩码的

#残差连接 + 层归一化（Add & Norm，第一次）
#解决深度网络的梯度消失问题，让梯度通过「残差边」直接回传，保证更深的网络能正常训练；同时保留原始输入信息，避免子层变换丢失语义。

#子层 2：Position-wise Feed-Forward Network（FFN，位置 - wise 前馈网络）
#对每个 token 独立做非线性特征变换，补全模型的非线性表达能力（注意力是纯线性操作，无 FFN 则模型只能拟合线性规律），
# 提取高级语义特征（如语法、情感、语义角色）。

#残差连接 + 层归一化（Add & Norm，第二次）

#注意：Encoder 的掩码（Mask）
# Encoder 自注意力是无掩码的，但如果输入序列有padding（批次内句子长度不同，补了<pad> token），需要加Padding Mask：
# 把<pad>位置的注意力权重置为-∞，经过 softmax 后权重趋近于 0，让模型不关注无效的 padding token。
# Mask 形状为 (B, 1, seq_len, seq_len)，在多头注意力中使用。

import torch
import torch.nn as nn
from .feed_forward import FeedForward as FFN
from .attention import MultiHeadAttention
from .positional_encoding import PositionalEncoding as PE

#单Encoder层
class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_head, d_ff, dropout=0.1):
        super().__init__()
        # 子层1：多头自注意力
        self.attn = MultiHeadAttention(d_model, n_head)
        # 子层2：FFN
        self.ffn = FFN(d_model, d_ff, dropout)
        # 两次层归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        # Dropout正则化
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # 子层1：多头自注意力 + 残差 + 层归一化
        attn_out,_ = self.attn(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attn_out))

        # 子层2：FFN + 残差 + 层归一化
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_out))

        return x


# Encoder(多层堆叠)
class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model=512, n_head=8, d_ff=2048, n_layers=6, max_len=5000, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        # 1. 词嵌入层
        self.embedding = nn.Embedding(vocab_size, d_model)
        # 2. 位置编码层
        self.pe = PE(d_model, max_len)

        # 3. 输入层Dropout
        self.dropout = nn.Dropout(dropout)
        # 4. 堆叠n_layers个Encoder层
        self.layers = nn.ModuleList([EncoderLayer(d_model, n_head, d_ff, dropout) for _ in range(n_layers)])
        # 最终层归一化
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        # x shape: (batch_size, seq_len)，输入为token索引
        batch_size, seq_len = x.size()

        # 输入：词嵌入 + 位置编码
        x = self.embedding(x) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        pe_tensor = self.pe(x)
        x = x + pe_tensor
        x = self.dropout(x)

        # 经过n_layers个Encoder层
        for layer in self.layers:
            x = layer(x, mask)

        # 最终层归一化
        return self.norm(x)

# 测试
if __name__ == '__main__':

    encoder = Encoder(vocab_size=10000, d_model=512, n_head=8, d_ff=2048, n_layers=6, max_len=500, dropout=0.1)
    #encoder参数量
    print("encoder参数量:", sum(p.numel() for p in encoder.parameters()))#encoder参数量: 24035328
    #测试前向传播
    x = torch.randint(0, 10000, (1, 10))#batch_size=1, seq_len=10
    output = encoder(x)
    print("encoder输出shape:", output.shape)#encoder输出shape: torch.Size([1, 10, 512])

    #3.测试一次的时间复杂度
    import time
    start = time.time()
    output = encoder(x)
    end = time.time()
    print("encoder一次的时间复杂度:", end-start)#encoder一次的时间复杂度: 0.01693248748779297
