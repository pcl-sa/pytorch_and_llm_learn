# 定义Transformer模型
import torch
import torch.nn as nn

# 导入你已实现的组件
from .encoder import Encoder
from .decoder import Decoder
from .positional_encoding import PositionalEncoding as PE


class Transformer(nn.Module):
    """
    Transformer 模型总装类
    整合：源词嵌入/位置编码 + 编码器栈 + 目标词嵌入/位置编码 + 解码器栈 + 输出层
    """
    def __init__(
        self,
        src_vocab_size: int,        # 源语言词表大小
        tgt_vocab_size: int,        # 目标语言词表大小
        d_model: int = 512,         # 模型维度
        n_head: int = 8,            # 多头注意力头数
        d_ff: int = 2048,           # 前馈网络中间维度
        n_layers: int = 6,          # Encoder/Decoder层数
        max_len: int = 500,         # 最大序列长度
        dropout: float = 0.1,        # Dropout概率
        pad_token_id: int = 0        # 填充tokenID，默认0
    ):
        super().__init__()
        self.d_model = d_model
        self.pad_token_id = pad_token_id        # 填充tokenID，默认0

        # ==================== 1. 词嵌入层 ====================
        # 源序列嵌入
        self.src_emb = nn.Embedding(src_vocab_size, d_model)
        # 目标序列嵌入
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model)

        # ==================== 2. 位置编码 ====================
        self.pos_encoding = PE(d_model, max_len)

        # ==================== 3. Dropout ====================
        self.dropout = nn.Dropout(dropout)

        # ==================== 4. 编码器 + 解码器 ====================
        self.encoder = Encoder(
            d_model=d_model,
            n_head=n_head,
            d_ff=d_ff,
            n_layers=n_layers,
            dropout=dropout
        )
        self.decoder = Decoder(
            d_model=d_model,
            n_head=n_head,
            d_ff=d_ff,
            n_layers=n_layers,
            dropout=dropout
        )

        # ==================== 5. 最终输出层 ====================
        # 将Decoder输出映射到目标词表，用于分类/生成
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)

    # ==================== 主类统一生成所有掩码（标准做法） ====================
    def _create_padding_mask(self, seq):
        # 输出：[B, 1, seq_len, seq_len]  方阵！适配Encoder自注意力
        batch_size, seq_len = seq.size()# [B, seq_len]
        mask = (seq != self.pad_token_id)  # [B, seq_len]
        mask = mask.unsqueeze(1).unsqueeze(2)  # [B,1,1,seq_len]
        mask = mask.expand(-1, -1, seq_len, -1)  # 广播为方阵 [B,1,seq_len,seq_len]
        return mask

    def _generate_causal_mask(self, tgt_len):
        mask = torch.tril(torch.ones(tgt_len, tgt_len, dtype=torch.bool))
        return mask.unsqueeze(0).unsqueeze(0)  # [1,1,T,T]

    def _create_cross_mask(self, src_seq, tgt_seq):
        # 1. 复用已有函数生成src的padding掩码 [B,1,src_len,src_len]
        src_mask = self._create_padding_mask(src_seq)

        # 2. 关键：提取src的有效掩码（去掉重复的行，变成 [B,1,1,src_len]）
        # 因为src_mask方阵每一行都一样，取第0行即可
        src_mask = src_mask[:, :, 0:1, :]

        # 3. 广播到目标序列长度 → 得到cross_mask标准形状
        tgt_len = tgt_seq.size(1)
        cross_mask = src_mask.expand(-1, -1, tgt_len, -1)

        return cross_mask

    def forward(self, src_seq: torch.Tensor, tgt_seq: torch.Tensor) -> torch.Tensor:
        """
        模型整体前向传播
        :param src_seq: 源序列 [batch_size, src_len]
        :param tgt_seq: 目标序列 [batch_size, tgt_len]
        :return: 模型输出 [batch_size, tgt_len, tgt_vocab_size]
        """
        # -------------------------- 1. 源序列编码 --------------------------
        # 词嵌入 + 缩放 + 位置编码 + Dropout
        device = src_seq.device
        src_len, tgt_len = src_seq.size(1), tgt_seq.size(1)
        src_emb = self.src_emb(src_seq) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        src_emb = self.pos_encoding(src_emb)

        src_emb = self.dropout(src_emb)
        src_mask = self._create_padding_mask(src_seq).to(device)

        # 编码器前向（自动生成padding掩码）
        enc_out = self.encoder(src_emb,src_mask)  # src_emb的形状: [batch_size, src_len, d_model], src_mask的形状: [batch_size, 1, src_len, src_len]
        # -------------------------- 2. 目标序列解码 --------------------------
        # 词嵌入 + 缩放 + 位置编码 + Dropout
        tgt_emb = self.tgt_emb(tgt_seq) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        tgt_emb = self.pos_encoding(tgt_emb)
        tgt_emb = self.dropout(tgt_emb)

        # 生成掩码
        tgt_padding_mask = self._create_padding_mask(tgt_seq).to(device)

        causal_mask = self._generate_causal_mask(tgt_len).to(device)

        combined_tgt_mask = tgt_padding_mask & causal_mask

        cross_mask = self._create_cross_mask(src_seq, tgt_seq).to(device)

        # 解码器前向（传入编码器输出+源序列，生成交叉掩码）
        dec_out = self.decoder(tgt_emb, enc_out, combined_tgt_mask, cross_mask)  # [batch_size, tgt_len, d_model]

        # -------------------------- 3. 最终输出 --------------------------
        output = self.fc_out(dec_out)
        # 训练时用CrossEntropyLoss，无需手动Softmax；推理时可打开
        # output = F.softmax(output, dim=-1)

        return output


# ==================== 测试代码 ====================
if __name__ == '__main__':
    # 超参数
    SRC_VOCAB_SIZE = 10000  # 源词表大小
    TGT_VOCAB_SIZE = 40000  # 目标词表大小
    BATCH_SIZE = 2
    SRC_LEN = 10 # 源序列长度
    TGT_LEN = 15 # 目标序列长度

    # 初始化Transformer模型
    model = Transformer(
        src_vocab_size=SRC_VOCAB_SIZE,
        tgt_vocab_size=TGT_VOCAB_SIZE
    )

    # 打印模型参数量
    print(f"Transformer总参数量: {sum(p.numel() for p in model.parameters()):,}")#Transformer总参数量: 90,258,496

    # 构造模拟输入
    src_seq = torch.randint(0, SRC_VOCAB_SIZE, (BATCH_SIZE, SRC_LEN))# 源序列, [2,10]
    tgt_seq = torch.randint(0, TGT_VOCAB_SIZE, (BATCH_SIZE, TGT_LEN))# 目标序列, [2,15]

    # 前向传播
    output = model(src_seq, tgt_seq)

    # 打印输出维度（验证正确性）
    print(f"源序列形状: {src_seq.shape}")#源序列形状: torch.Size([2, 4])
    print(f"目标序列形状: {tgt_seq.shape}")#目标序列形状: torch.Size([2, 2])
    print(f"模型输出形状: {output.shape}")#模型输出形状: torch.Size([2, 2, 40000])

    #一次前向传播时间复杂度
    import time
    start = time.time()
    output = model(src_seq, tgt_seq)
    end = time.time()
    print(f"一次前向传播时间: {end - start:.4f} 秒")#一次前向传播时间: 0.0481 秒
