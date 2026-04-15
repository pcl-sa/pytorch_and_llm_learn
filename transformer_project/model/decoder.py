# 解码器实现
#输入：已经生成的目标序列前缀 + Encoder 输出的源序列全局上下文
#输出：下一个词的概率分布
#一步步生成目标序列，每一步都只能用「已经生成的历史信息」，
#绝对不能看到「未来还没生成的词」，严格保证自回归特性
# Decoder 的 6 个核心设计
#1.堆叠结构：和 Encoder 一样
#2.子层数量：比 Encoder 多 1 个子层，共 3 个核心子层 / 层
#3.新增子层：Encoder-Decoder 交叉注意力，用于融合 Encoder 的源序列上下文
#4.残差 + 层归一化：每个子层后都有，和 Encoder 完全一致
#5.掩码自注意力：对自注意力加因果掩码，防止看到未来位置，保证自回归
#6.输入偏移：目标序列嵌入右移一位，保证预测i时仅用1~i-1的已知输出

# 子层 1：Masked Multi-Head Self-Attention（带掩码的多头自注意力）
# 这是 Decoder 和 Encoder 自注意力的核心区别，也是自回归生成的保障。
# 掩码矩阵是下三角矩阵：对角线及以下为 1（允许关注），以上为 0（屏蔽）

# 子层 2：Encoder-Decoder Cross-Attention（编码器 - 解码器交叉注意力）
# 让 Decoder 在生成每个目标词时，主动关注源序列中最相关的部分，实现源语言和目标语言的语义对齐。

# 子层 3：Position-wise Feed-Forward Network（位置 - wise 前馈网络）
# 和 Encoder 的 FFN完全一致，没有任何区别！

#残差连接 + 层归一化（Add & Norm）每个子层之后都有残差连接 + 层归一化

import torch
import torch.nn as nn
from .feed_forward import FeedForward as FFN
from .attention import MultiHeadAttention

class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_head, d_ff, dropout=0.1):
        super().__init__()
        # 子层1：带掩码的多头自注意力
        self.self_attn = MultiHeadAttention(d_model, n_head)
        # 子层2：Encoder-Decoder交叉注意力
        self.cross_attn = MultiHeadAttention(d_model, n_head)
        # 子层3：FFN
        self.ffn = FFN(d_model, d_ff, dropout)
        # 3组层归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        # Dropout正则化
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, enc_out, cross_mask=None, causal_mask=None):
        # 子层1：Masked自注意力 + 残差 + 层归一化
        attn_out, _ = self.self_attn(x, x, x, causal_mask)
        x = self.norm1(x + self.dropout1(attn_out))

        # 子层2：交叉注意力 + 残差 + 层归一化（Q来自x，K/V来自enc_out）
        cross_attn_out, _ = self.cross_attn(x, enc_out, enc_out, cross_mask)
        x = self.norm2(x + self.dropout2(cross_attn_out))

        # 子层3：FFN + 残差 + 层归一化
        ffn_out = self.ffn(x)
        x = self.norm3(x + self.dropout3(ffn_out))

        return x

# -------------------------- 3. 完整Decoder（N层堆叠） --------------------------
class Decoder(nn.Module):
    def __init__(self, d_model=512, n_head=8, d_ff=2048, n_layers=6, dropout=0.1):
        super().__init__()
        # 堆叠n_layers个Decoder层
        self.layers = nn.ModuleList([DecoderLayer(d_model, n_head, d_ff, dropout) for _ in range(n_layers)])

    def forward(self, x, enc_out, causal_mask=None, cross_mask=None):
        """
        x: 主类处理好的目标嵌入张量 [batch, tgt_len, d_model]
        enc_out: 编码器输出 [batch, src_len, d_model]
        causal_mask: 因果掩码（主类传入）
        cross_mask: 交叉注意力掩码（主类传入）
        """
        for layer in self.layers:
            x = layer(x, enc_out, cross_mask, causal_mask)
        return x