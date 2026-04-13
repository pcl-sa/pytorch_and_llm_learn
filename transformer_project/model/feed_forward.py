# 初始化前馈网络
#核心职责：实现 Encoder/Decoder 共用的Position-wise Feed-Forward Network，包含两层线性变换 + ReLU 激活，参数按层独立。
#核心作用：1.为每个位置的输入提供非线性变换，增加模型的表达能力。
#        2.对注意力融合后的特征做「深加工」
#        3.升维 - 降维的瓶颈结构，高效提取特征
#        4.等价于 1×1 卷积，提取通道维度特征
#        5.保证并行性，支撑 Transformer 的高效训练
from torch import nn
import torch.nn.functional as F
import torch

class FeedForward(nn.Module):
    def __init__(self,d_model:int,d_ff:int=None,dropout:float=0.1):
        super(FeedForward, self).__init__()
        if d_ff is None:
            d_ff = d_model*4
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 公式：FFN(x) = max(0, xW1 + b1)W2 + b2
        # x shape: (B, seq_len, d_model)
        x = self.linear1(x)  # (B, seq_len, d_ff)
        x = F.relu(x)   # ReLU激活，引入非线性
        x = self.dropout(x)  # dropout正则化
        x = self.linear2(x)  # (B, seq_len, d_model)
        return x

# 测试前馈网络
if __name__ == "__main__":
    ffn = FeedForward(d_model=1024,dropout=0.1)# 初始化前馈网络
    #ffn的参数量
    print("ffn的参数量:", sum(p.numel() for p in ffn.parameters()))#ffn的参数量: 8393728
    #ffn的前向传播
    x = torch.randn(1, 10, 1024)  # (B, seq_len, d_model)
    output = ffn(x)
    print("ffn的输出形状:", output.shape)

    ##3.测试一次的时间复杂度
    import time
    start = time.time()
    output = ffn(x)
    end = time.time()
    print("ffn一次的时间复杂度:", end-start)#ffn一次的时间复杂度: 0.002777099609375
