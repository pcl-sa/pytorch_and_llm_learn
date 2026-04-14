# 初始化文本转换模块
#用于将输入的文本序列转换为模型可以处理的表示
#既文本序列转换成向量序列
from model.word2index import Word2Idx


# # 训练单词变换模块
# #训练中文语料库
# # 词汇表大小：8000 是最优值（小语料+手写模型完美适配）
VOCAB_SIZE = 8000
# 最大句子长度：512 完全够用
MAX_SEQ_LEN = 512

# 初始化单词变换模块
word2idx = Word2Idx()

# ======================
# 1. 训练中文 SentencePiece 模型
# ======================
print("开始训练中文分词模型...")
word2idx.train_model(
    train_file="../data/Chinese_English_parallel_corpus/chinese.txt",
    model_prefix="../data/Chinese_English_parallel_corpus/Chinese",
    model_type='bpe',
    vocab_size=VOCAB_SIZE,
    max_sentence_length=MAX_SEQ_LEN,
    character_coverage=1.0,  # 中文必须用1.0
)

# 加载中文模型 → 打印词汇统计
word2idx.load_model("../data/Chinese_English_parallel_corpus/Chinese.model")
print("中文模型词汇统计：", word2idx.sp.get_piece_size())

# ======================
# 2. 训练英文 SentencePiece 模型
# ======================
print("\n开始训练英文分词模型...")
word2idx.train_model(
    train_file="../data/Chinese_English_parallel_corpus/english.txt",
    model_prefix="../data/Chinese_English_parallel_corpus/English",
    model_type='bpe',
    vocab_size=VOCAB_SIZE,
    max_sentence_length=MAX_SEQ_LEN,
    character_coverage=0.9995,  # 英文专用参数
)

# 加载英文模型 → 打印词汇统计
word2idx.load_model("../data/Chinese_English_parallel_corpus/English.model")
print("英文模型词汇统计：", word2idx.sp.get_piece_size())

print("\n✅ 双语模型训练完成！")
