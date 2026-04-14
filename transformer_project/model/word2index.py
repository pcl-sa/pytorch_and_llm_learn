# word_transformation/word2idx.py
import sentencepiece as spm
from typing import List, Optional
import os


class Word2Idx:
    def __init__(self):
        self.sp = spm.SentencePieceProcessor()
        self._is_loaded = False

    def train_model(self, train_file: str, model_prefix: str, vocab_size: int,
                    model_type: str = "bpe", max_sentence_length: int = 512,
                    character_coverage: float = 1.0, pad_id: int = 0,
                    unk_id: int = 1, bos_id: int = 2, eos_id: int = 3,
                    num_threads: int = 16, verbose: bool = False) -> None:
        """
        训练SentencePiece模型

        Args:
            train_file: 训练文本文件路径
            model_prefix: 模型保存前缀
            vocab_size: 词汇表大小
            model_type: 模型类型 (bpe, unigram, char, word)
            max_sentence_length: 最大句子长度
            character_coverage: 字符覆盖率（中文建议1.0）
            pad_id/pad_id/eos_id/unk_id: 特殊token的ID
            num_threads: 训练线程数
            verbose: 是否显示训练日志
        """
        # 检查输入文件
        if not os.path.exists(train_file):
            raise FileNotFoundError(f"训练文件不存在: {train_file}")

        # 训练参数
        train_args = f"""
            --input={train_file}
            --model_prefix={model_prefix}
            --vocab_size={vocab_size}
            --model_type={model_type}
            --max_sentence_length={max_sentence_length}
            --character_coverage={character_coverage}
            --pad_id={pad_id}
            --unk_id={unk_id}
            --bos_id={bos_id}
            --eos_id={eos_id}
            --control_symbols=<pad>,<bos>,<eos>
            --bos_piece=<bos>
            --eos_piece=<eos>
            --unk_piece=<unk>
            --pad_piece=<pad>
            --split_by_whitespace=false
            --num_threads={num_threads}
        """

        if not verbose:
            train_args += " --log_level=error"

        spm.SentencePieceTrainer.train(train_args)
        print(f"✅ 模型训练完成，保存为: {model_prefix}.model")

    def load_model(self, model_path: str) -> None:
        """加载预训练模型"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")

        self.sp.Load(model_path)
        self._is_loaded = True
        print(f"✅ 模型加载成功，词汇表大小: {self.sp.get_piece_size()}")

    def _ensure_loaded(self) -> None:
        """确保模型已加载"""
        if not self._is_loaded:
            raise RuntimeError("模型未加载，请先调用 load_model()")

    def text_to_ids(self, text: str, max_len: int = 512,
                    add_bos: bool = False, add_eos: bool = False) -> List[int]:
        """
        文本转ID序列

        Args:
            text: 输入文本
            max_len: 最大长度
            add_bos: 是否添加BOS token
            add_eos: 是否添加EOS token

        Returns:
            ID列表（长度固定为max_len）
        """
        self._ensure_loaded()

        if not text:
            ids = []
        else:
            ids = self.sp.encode(text)

        # 添加特殊token
        if add_bos:
            ids = [self.sp.bos_id()] + ids
        if add_eos:
            ids = ids + [self.sp.eos_id()]

        # 截断
        if len(ids) > max_len:
            ids = ids[:max_len]
            if add_eos:  # 如果截断了，确保最后一个不是截断的eos
                ids[-1] = self.sp.eos_id()

        # 填充
        padding_length = max_len - len(ids)
        if padding_length > 0:
            ids += [self.sp.pad_id()] * padding_length

        return ids

    def ids_to_text(self, ids: List[int],
                    remove_special: bool = True,
                    ignore_ids: Optional[List[int]] = None) -> str:
        """
        ID序列转文本

        Args:
            ids: ID列表
            remove_special: 是否移除特殊token
            ignore_ids: 额外需要忽略的ID列表

        Returns:
            解码后的文本
        """
        self._ensure_loaded()

        if remove_special:
            # 默认忽略的特殊ID
            special_ids = {self.sp.pad_id(), self.sp.unk_id(),
                           self.sp.bos_id(), self.sp.eos_id()}
            if ignore_ids:
                special_ids.update(ignore_ids)

            filtered_ids = [i for i in ids if i not in special_ids]
        else:
            filtered_ids = ids

        return self.sp.decode(filtered_ids) if filtered_ids else ""

    def get_vocab_size(self) -> int:
        """获取词汇表大小"""
        self._ensure_loaded()
        return self.sp.get_piece_size()

    def get_id_from_token(self, token: str) -> int:
        """获取token对应的ID"""
        self._ensure_loaded()
        return self.sp.piece_to_id(token)

    def get_token_from_id(self, token_id: int) -> str:
        """获取ID对应的token"""
        self._ensure_loaded()
        return self.sp.id_to_piece(token_id)

    def batch_text_to_ids(self, texts: List[str], max_len: int = 512,
                          add_bos: bool = False, add_eos: bool = False) -> List[List[int]]:
        """批量文本转ID"""
        return [self.text_to_ids(text, max_len, add_bos, add_eos) for text in texts]

    def batch_ids_to_text(self, batch_ids: List[List[int]],
                          remove_special: bool = True) -> List[str]:
        """批量ID转文本"""
        return [self.ids_to_text(ids, remove_special) for ids in batch_ids]


# 使用示例
if __name__ == "__main__":
    # 初始化
    tokenizer = Word2Idx()

    # 训练模型
    # tokenizer.train_model("train.txt", "my_model", vocab_size=8000)

    # 加载模型
    tokenizer.load_model("../../data/Chinese_English_parallel_corpus/Chinese.model")

    # 编码解码
    text = "这是一个测试句子"
    ids = tokenizer.text_to_ids(text, max_len=10, add_bos=True, add_eos=True)
    print(f"编码: {ids}")

    decoded = tokenizer.ids_to_text(ids)
    print(f"解码: {decoded}")

    # 批量处理
    texts = ["句子1", "句子2"]
    batch_ids = tokenizer.batch_text_to_ids(texts, max_len=5)
    print(f"批量编码: {batch_ids}")