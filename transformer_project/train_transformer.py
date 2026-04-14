# 训练Transformer模型

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import sentencepiece as spm

# ====================== 固定配置 ======================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VOCAB_SIZE = 8000          # 分词模型词汇表
MAX_LEN = 512             # 最大序列长度
BATCH_SIZE = 2            # 手写模型用小批次
EPOCHS = 10               # 训练轮数
D_MODEL = 512             # Transformer嵌入维度（和你模型一致）

# 初始化中英分词器
zh_tokenizer = spm.SentencePieceProcessor("../data/Chinese_English_parallel_corpus/Chinese.model")
en_tokenizer = spm.SentencePieceProcessor("../data/Chinese_English_parallel_corpus/English.model")



class TranslationDataset(Dataset):
    def __init__(self, zh_path, en_path):
        # 读取平行语料（逐行对应）
        with open(zh_path, 'r', encoding='utf-8') as f:
            self.zh_sentences = [line.strip() for line in f.readlines()]
        with open(en_path, 'r', encoding='utf-8') as f:
            self.en_sentences = [line.strip() for line in f.readlines()]

    def __len__(self):
        return len(self.zh_sentences)

    def __getitem__(self, idx):
        # 1. 中文 → 编码器输入 (src)
        src = zh_tokenizer.word2idx(self.zh_sentences[idx])
        # 2. 英文 → 解码器输入 (tgt_input) + 标签 (tgt_label)
        tgt = en_tokenizer.word2idx(self.en_sentences[idx])

        # 解码器输入：右移一位（去掉EOS，前面加BOS）
        tgt_input = [2] + tgt[:-1]
        # 标签：原始英文索引（模型要预测的结果）
        tgt_label = tgt

        return (
            torch.tensor(src, dtype=torch.long).to(DEVICE),
            torch.tensor(tgt_input, dtype=torch.long).to(DEVICE),
            torch.tensor(tgt_label, dtype=torch.long).to(DEVICE)
        )


# 加载数据集
dataset = TranslationDataset(
    zh_path="../data/Chinese_English_parallel_corpus/chinese.txt",
    en_path="../data/Chinese_English_parallel_corpus/english.txt"
)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# 这里导入你自己写的 Transformer 模型！
from model import Transformer

# 初始化模型（词汇表对齐分词模型）
model = Transformer(
    src_vocab_size=VOCAB_SIZE,   # 中文词汇表
    tgt_vocab_size=VOCAB_SIZE,   # 英文词汇表
    d_model=D_MODEL,
    max_len=MAX_LEN
).to(DEVICE)

# 损失函数：忽略<PAD>（0）的损失，这是翻译模型必须加的！
criterion = nn.CrossEntropyLoss(ignore_index=0)

# 优化器：Transformer论文原版Adam
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

print("🚀 开始训练 Transformer 翻译模型...")
model.train()  # 开启训练模式

for epoch in range(EPOCHS):
    total_loss = 0
    for src, tgt_input, tgt_label in dataloader:
        optimizer.zero_grad()  # 清空梯度

        # 🔥 前向传播：输入你的手写Transformer
        output = model(src, tgt_input)

        # 维度调整（适配损失函数）
        output = output.reshape(-1, VOCAB_SIZE)
        tgt_label = tgt_label.reshape(-1)

        # 计算损失
        loss = criterion(output, tgt_label)

        # 反向传播 + 更新参数
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # 打印训练结果
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch [{epoch + 1}/{EPOCHS}] | 平均损失: {avg_loss:.4f}")

# 保存训练好的模型
torch.save(model.state_dict(), "translation_transformer.pth")
print("✅ 模型训练完成，已保存！")


def translate(text):
    model.eval()
    with torch.no_grad():
        # 中文→索引
        src = zh_tokenizer.text_to_ids(text, MAX_LEN)
        src = torch.tensor([src], dtype=torch.long).to(DEVICE)

        # 初始化解码器输入（以<BOS>开头）
        tgt_input = torch.tensor([[2]], dtype=torch.long).to(DEVICE)

        # 生成翻译
        for _ in range(MAX_LEN):
            output = model(src, tgt_input)
            # 取最后一个词的预测结果
            next_token = output.argmax(-1)[:, -1]
            tgt_input = torch.cat([tgt_input, next_token.unsqueeze(0)], dim=1)

            # 遇到<EOS>停止
            if next_token.item() == 3:
                break

        # 索引→英文
        result = en_tokenizer.ids_to_text(tgt_input.squeeze().tolist())
        return result


# 测试翻译
print(translate("我爱机器学习"))
