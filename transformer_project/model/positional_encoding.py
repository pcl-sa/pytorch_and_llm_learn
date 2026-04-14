#这是一个位置编码层的实现
# 用于将序列中的位置信息编码到模型中
# 基于pytorch框架实现
import torch
from torch import nn
import math


#核心职责：实现 Transformer 专用的正弦 / 余弦位置编码，
#为无循环、无卷积的模型注入序列位置信息，支持与词嵌入向量直接相加。

class PositionalEncoding(nn.Module):
    def __init__(self,d_model:int,max_len:int=5000):
        '''
        初始化位置编码层
        :param d_model: 模型的维度，必须是偶数
        :param max_len: 最大序列长度，默认5000
        '''
        super().__init__()
        # 1. 预计算位置编码矩阵（固定值，无需优化）
        pe=torch.zeros(max_len,d_model)
        position=torch.arange(0,max_len,dtype=torch.float).unsqueeze(1)
        div_term=torch.exp(torch.arange(0,d_model,2).float()*(-math.log(10000.0)/d_model))

        #偶数维度用sin，奇数维度用cos
        pe[:,0::2] = torch.sin(position*div_term)
        pe[:,1::2] = torch.cos(position*div_term)

        # 2. 关键：用register_buffer注册（参数名，张量）
        # 第二个参数设为False，表示该张量不需要梯度（默认也是False）
        self.register_buffer('pe', pe.unsqueeze(0))#(1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch_size, seq_len, d_model]
        # 取出对应长度的位置编码并加到输入上
        # x shape: [batch_size, seq_len, d_model]
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return x


# 测试位置编码层
if __name__ == '__main__':
    # 初始化位置编码层（d_model=512，最大序列长度5000）
    pe_layer = PositionalEncoding(d_model=512)
    # 1. 验证buffer会被存入state_dict
    print("pe是否在state_dict中:", 'pe' in pe_layer.state_dict())  # 输出 True

    # 2. 验证不是可训练参数
    params = list(pe_layer.parameters())
    print("可训练参数数量:", len(params))  # 输出 0（无参数）

    # 3. 前向传播示例
    x = torch.randn(2, 100, 512).to('cpu')  # batch=2, seq_len=100, d_model=512
    output = pe_layer(x)
    print("输出shape:", output.shape)  # 输出 torch.Size([2, 100, 512])

    #验证位置编码一次的时间复杂度
    import time
    x_2=torch.randn(2, 100, 1024).to('cpu')
    p_layer = PositionalEncoding(d_model=1024)
    start = time.time()
    output = p_layer(x_2)
    end = time.time()
    print("位置编码一次的时间复杂度:", end-start)#本机器：位置编码一次的时间复杂度: 0.0002472400665283203

