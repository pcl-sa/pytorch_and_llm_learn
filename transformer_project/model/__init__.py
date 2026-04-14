#这是一个transformer模型的核心代码实现
# 包含了transformer模型的所有组件，如编码器、解码器、注意力机制等
# 基于pytorch框架实现

#只向外提供transformer模型,以及Encoder和Decoder组件
from .encoder import EncoderLayer
from .decoder import DecoderLayer
from .transformer import Transformer, Encoder, Decoder
from .word2index import Word2Idx
