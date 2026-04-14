# 这是一个自注意力机制的实现
# 包含了头自注意力机制、多头自注意力机制等
# 基于pytorch框架实现

import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaleDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()  # 调用父类构造函数

    def forward(self, Q, K, V, mask=None):
        # ========== 步骤1：计算 Q 和 K 的点积（相似度） ==========
        # Q形状：(B, n_head, seq_len_q, d_k)  （多头场景）或 (B, seq_len_q, d_k)（单头）
        # K形状：(B, n_head, seq_len_k, d_k)
        # K.transpose(-2, -1)：把K的最后两个维度交换，变成 (B, n_head, d_k, seq_len_k)
        # torch.matmul：矩阵乘法，结果scores形状：(B, n_head, seq_len_q, seq_len_k)
        # 物理意义：seq_len_q 个查询，每个查询和 seq_len_k 个键计算相似度得分
        scores = torch.matmul(Q, K.transpose(-2, -1))

        # ========== 步骤2：缩放（核心优化，解决梯度消失） ==========
        # d_k = Q最后一维的长度（每个头的维度），比如64
        d_k = Q.size(-1)
        # 除以 sqrt(d_k)：将点积结果的方差从d_k缩放到1，避免softmax饱和
        # 注意：torch.sqrt里必须转float32，否则如果Q是float16会报错
        scores = scores / torch.sqrt(torch.tensor(d_k, dtype=torch.float32,device=Q.device))

        # ========== 步骤3：应用掩码（可选，解码器核心） ==========
        # mask形状：(B, n_head, seq_len_q, seq_len_k) 或 (B, 1, seq_len_q, seq_len_k)
        # mask==0的位置：是需要屏蔽的位置（比如未来的词、padding的词）
        # masked_fill：把mask==0的位置的scores设为-1e9（极小值），softmax后权重≈0
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # ========== 步骤4：Softmax归一化，得到注意力权重 ==========
        # dim=-1：对最后一维（seq_len_k）做softmax，保证每个查询的权重和为1
        # attn_weights形状：和scores完全一致 (B, n_head, seq_len_q, seq_len_k)
        # 物理意义：每个查询对每个键的注意力权重
        attn_weights = F.softmax(scores, dim=-1)

        # ========== 步骤5：加权求和，得到最终上下文向量 ==========
        # V形状：(B, n_head, seq_len_k, d_k)
        # matmul(权重, V)：每个查询的上下文向量 = 所有键的值 * 对应权重 求和
        # output形状：(B, n_head, seq_len_q, d_k)
        output = torch.matmul(attn_weights, V)

        # 返回上下文向量 + 注意力权重（权重用于可视化/分析）
        return output, attn_weights


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()
        assert d_model % n_head == 0,"d_model必须能被n_head整除"
        # 保存超参数：头数 + 每个头的维度（必须能被d_model整除）
        self.d_model = d_model
        self.n_head = n_head  # 比如8
        self.d_k = d_model // n_head  # 比如512//8=64

        # ========== 定义线性投影层（论文中的W^Q/W^K/W^V） ==========
        # 输入维度d_model，输出维度d_model（因为要拆分成n_head个d_k）
        # 比如：w_q是 (512, 512) 的矩阵，作用是把输入的Q投影到多头空间
        self.w_q = nn.Linear(d_model, d_model)  # Q的投影层
        self.w_k = nn.Linear(d_model, d_model)  # K的投影层
        self.w_v = nn.Linear(d_model, d_model)  # V的投影层

        # ========== 定义最终输出投影层（论文中的W^O） ==========
        # 把拼接后的多头结果投影回d_model维度
        self.w_o = nn.Linear(d_model, d_model)

        # ========== 实例化缩放点积注意力 ==========
        self.attention = ScaleDotProductAttention()

    def forward(self, q, k, v, mask=None):
        # 输入形状：q/k/v都是 (batch_size, n_head, seq_len, d_model)
        # mask形状：(B, seq_len_q, seq_len_k)（如果有）

        # 保存batch size，后续reshape要用
        batch_size = q.size(0)
        seq_len = q.size(1)

        # ========== 步骤1：线性投影（Q/K/V分别过投影层） ==========
        # Q/K/V形状：(B, seq_len, d_model)
        # 物理意义：把原始嵌入向量投影到“注意力空间”，为拆分多头做准备
        Q = self.w_q(q)
        K = self.w_k(k)
        V = self.w_v(v)

        # ========== 步骤2：拆分多头（核心操作） ==========
        # 目标：把 (B, seq_len, d_model) → (B, n_head, seq_len, d_k)
        # 分两步：view拆分维度 → transpose交换维度顺序
        # 1. view：(B, seq_len, n_head, d_k) → 把d_model拆成n_head*d_k
        # 2. transpose(1,2)：把n_head维度换到第二个位置，方便并行计算
        Q = Q.view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)
        # 此时Q/K/V形状：(B, n_head, seq_len, d_k)

        # ========== 步骤3：对每个头并行计算缩放点积注意力 ==========
        x, attn = self.attention(Q, K, V, mask)

        # ========== 步骤4：拼接多头结果（核心操作） ==========
        # 目标：把 (B, n_head, seq_len_q, d_k) → (B, seq_len_q, d_model)
        # 分两步：transpose交换维度 → contiguous+view拼接
        # 1. transpose(1,2)：把n_head维度换回第三个位置 → (B, seq_len_q, n_head, d_k)
        # 2. contiguous()：保证内存连续（transpose后内存不连续，view会报错）
        # 3. view：把n_head*d_k合并回d_model → (B, seq_len_q, d_model)
        x = x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        # ========== 步骤5：最终线性投影 ==========
        # 把拼接后的结果过W^O层，投影回d_model维度
        # output形状：(B, seq_len_q, d_model)
        output = self.w_o(x)

        # 返回最终输出（注意力权重可选返回，用于分析）
        return output, attn

# 测试多头注意力机制、
if __name__ == '__main__':
    # 初始化多头注意力机制（d_model=512，n_head=8）
    attn_layer = MultiHeadAttention(d_model=1024, n_head=8)
    #1.查看模型参数量
    print("模型参数量:", sum(p.numel() for p in attn_layer.parameters()))  # 模型参数量: 4198400
    #2.前向传播示例
    x = torch.randn(2, 100, 1024).to('cpu')  # batch=2, seq_len=100, d_model=1024
    output,_ = attn_layer(x, x, x)
    print("输出shape:", output.shape)  # 输出 torch.Size([2, 100, 1024])
    #3.测试一次的时间复杂度
    import time
    x_2=torch.randn(2, 100, 1024).to('cpu')
    attn_layer = MultiHeadAttention(d_model=1024, n_head=8)
    start = time.time()
    output,_ = attn_layer(x_2, x_2, x_2)
    end = time.time()
    print("多头注意力机制一次的时间复杂度:", end-start)#本机器：多头注意力机制一次的时间复杂度: 0.008917570114135742

