# 定义Transformer模型
import torch
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
        dropout: float = 0.1        # Dropout概率
    ):
        super().__init__()
        self.d_model = d_model

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
            vocab_size=src_vocab_size,
            d_model=d_model,
            n_head=n_head,
            d_ff=d_ff,
            n_layers=n_layers,
            max_len=max_len,
            dropout=dropout
        )
        self.decoder = Decoder(
            vocab_size=tgt_vocab_size,
            d_model=d_model,
            n_head=n_head,
            d_ff=d_ff,
            n_layers=n_layers,
            max_len=max_len,
            dropout=dropout
        )

        # ==================== 5. 最终输出层 ====================
        # 将Decoder输出映射到目标词表，用于分类/生成
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src_seq: torch.Tensor, tgt_seq: torch.Tensor) -> torch.Tensor:
        """
        模型整体前向传播
        :param src_seq: 源序列 [batch_size, src_len]
        :param tgt_seq: 目标序列 [batch_size, tgt_len]
        :return: 模型输出 [batch_size, tgt_len, tgt_vocab_size]
        """
        # -------------------------- 1. 源序列编码 --------------------------
        # 词嵌入 + 缩放 + 位置编码 + Dropout
        src_emb = self.src_emb(src_seq) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        src_emb = self.pos_encoding(src_emb)
        src_emb = self.dropout(src_emb)

        # 编码器前向（自动生成padding掩码）
        enc_out = self.encoder(src_seq)  # [batch_size, src_len, d_model]

        # -------------------------- 2. 目标序列解码 --------------------------
        # 词嵌入 + 缩放 + 位置编码 + Dropout
        tgt_emb = self.tgt_emb(tgt_seq) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        tgt_emb = self.pos_encoding(tgt_emb)
        tgt_emb = self.dropout(tgt_emb)

        # 解码器前向（传入编码器输出+源序列，生成交叉掩码）
        dec_out = self.decoder(tgt_seq, enc_out, src_seq)  # [batch_size, tgt_len, d_model]

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
    print(f"Transformer总参数量: {sum(p.numel() for p in model.parameters()):,}")#Transformer总参数量: 115,860,544

    # 构造模拟输入
    src_seq = torch.randint(0, SRC_VOCAB_SIZE, (BATCH_SIZE, SRC_LEN))# 源序列, [batch_size, src_len]
    tgt_seq = torch.randint(0, TGT_VOCAB_SIZE, (BATCH_SIZE, TGT_LEN))# 目标序列, [batch_size, tgt_len]

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
    print(f"一次前向传播时间: {end - start:.4f} 秒")#一次前向传播时间: 0.0290 秒
