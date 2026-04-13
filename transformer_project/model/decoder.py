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
from .positional_encoding import PositionalEncoding as PE

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
    def __init__(self, vocab_size, d_model=512, n_head=8, d_ff=2048, n_layers=6, max_len=500, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, d_model)
        # 位置编码
        self.pe = PE(d_model, max_len)
        # Dropout
        self.dropout = nn.Dropout(dropout)
        # 堆叠n_layers个Decoder层
        self.layers = nn.ModuleList([DecoderLayer(d_model, n_head, d_ff, dropout) for _ in range(n_layers)])
        # 最终层归一化
        self.norm = nn.LayerNorm(d_model)

    def create_padding_mask(self, seq, pad_token_id=0):
        """
        生成Padding掩码
        :param seq: 输入序列，形状 (batch_size, seq_len)
        :param pad_token_id: Padding的token索引，默认0
        :return: padding_mask，形状 (batch_size, 1, 1, seq_len)
        """
        # 先判断哪些位置是Padding（seq == pad_token_id）
        mask = (seq != pad_token_id).unsqueeze(1).unsqueeze(2)
        # 形状变化：(batch_size, seq_len) → (batch_size, 1, 1, seq_len)
        return mask

    def create_cross_attention_mask(self, src_seq, tgt_seq, pad_token_id=0):
        """
        生成交叉注意力的非方阵掩码（合并源+目标的Padding掩码）
        :param src_seq: 源序列 (batch_size, src_len)
        :param tgt_seq: 目标序列 (batch_size, tgt_len)
        :return: cross_mask，形状 (batch_size, 1, tgt_len, src_len)
        """
        # 1. 生成源Padding掩码 (batch_size, 1, 1, src_len)
        src_mask = self.create_padding_mask(src_seq, pad_token_id)
        # 2. 生成目标Padding掩码 (batch_size, 1, 1, tgt_len)
        tgt_mask = self.create_padding_mask(tgt_seq, pad_token_id)
        # 2. 转换为 (batch_size, 1, tgt_len, 1) 形状
        tgt_mask = tgt_mask.permute(0, 1, 3, 2)
        # 3. 广播合并 → 形状 (batch_size, 1, tgt_len, src_len)
        cross_mask = src_mask & tgt_mask
        return cross_mask

    def generate_causal_mask(self, tgt_len):
        """生成因果掩码（下三角矩阵）"""
        mask = torch.tril(torch.ones(tgt_len, tgt_len, dtype=torch.bool))
        # 扩展维度：(T,T) → (1,1,T,T)，适配批量+多头
        mask = mask.unsqueeze(0).unsqueeze(0)
        return mask

    def forward(self, tgt_x, enc_out, src_seq, pad_token_id=0):
        '''
        Decoder前向传播
        :param tgt_x: 目标序列，形状 (batch_size, tgt_len)
        :param enc_out: Encoder输出，形状 (batch_size, src_len, d_model)
        :param src_seq: 源序列，形状 (batch_size, src_len)
        :param pad_token_id: Padding的token索引，默认0
        :return: Decoder输出，形状 (batch_size, tgt_len, d_model)
        '''
        batch_size, tgt_len = tgt_x.size()
        src_len=enc_out.size(1)

        # 1. 词嵌入 + 位置编码
        x = self.embedding(tgt_x) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        x = x + self.pe(x)
        x = self.dropout(x)

        # 2. 生成因果掩码（tgt_mask）
        causal_mask = self.generate_causal_mask(tgt_len).to(x.device)
        # 3. 生成交叉注意力掩码（src_mask）
        cross_mask=self.create_cross_attention_mask(src_seq, tgt_x, pad_token_id).to(x.device)

        # 3. 经过n_layers个Decoder层
        for layer in self.layers:
            x = layer(x, enc_out, cross_mask, causal_mask)

        # 4. 最终层归一化
        return self.norm(x)

#测试
if __name__ == '__main__':
    decoder = Decoder(vocab_size=10000, d_model=512, n_head=8, d_ff=2048, n_layers=6, max_len=500, dropout=0.1)
    #decoder参数量
    print("decoder参数量:", sum(p.numel() for p in decoder.parameters()))#decoder参数量: 30345216
    # 模拟输入：机器翻译场景
    pad_token_id = 0
    # 源序列（batch_size=2, src_len=4）：["我","爱","中国","<PAD>"], ["你","好","吗","<PAD>"]
    src_seq = torch.tensor([[1, 2, 3, 0], [4, 5, 6, 0]])
    # 目标序列（batch_size=2, tgt_len=2）：["I","love"], ["Hi","there"]
    tgt_seq = torch.tensor([[7, 8], [9, 10]])
    # Encoder
    from .encoder import Encoder
    encoder=Encoder(vocab_size=10000, d_model=512, n_head=8, d_ff=2048, n_layers=6, max_len=500, dropout=0.1)
    enc_out = encoder(src_seq)
    print("encoder输出shape:", enc_out.shape)#torch.Size([2, 4, 512])
    # Decoder
    output = decoder(tgt_seq, enc_out, src_seq, pad_token_id=pad_token_id)
    print("decoder输出shape:", output.shape)
    cross_mask = decoder.create_cross_attention_mask(src_seq, tgt_seq, pad_token_id)
    print("非方阵掩码形状：", cross_mask.shape)
