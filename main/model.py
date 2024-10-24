import torch
from torch import nn
import math
from typing import Tuple
import config


class PositionalEncoding(torch.nn.Module):
    """
    位置编码类
    """
    def __init__(self, emb_size: int, dropout: float, maxlen: int = 500):
        """
        emb_size:嵌入向量的维度
        dropout:防止过拟合
        maxlen:句子最大长度
        """
        super(PositionalEncoding, self).__init__()
        # 公式
        den = torch.exp(-torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        # 位置向量
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        # 偶数位置使用sin，奇数位置使用cos
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        # 位置向量扩展一个维度
        pos_embedding = pos_embedding.unsqueeze(-2)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("pos_embedding", pos_embedding)

    def forward(self, token_embedding):
        return self.dropout(
            token_embedding + self.pos_embedding[: token_embedding.size(0), :]
        )

# 模型
class Txt2SqlTransformer(nn.Module):

    def __init__(
        self, vocab_size, embed_dim=512, d_model=512, num_layers=4, n_head=16
    ):
        """
        vocab_size:词汇表的大小(词的种类)，不同的模型的词汇表不同，切换模型实例化时注意！
        embed_dim:词向量的维度
        d_model:transformer中编码器和解码器中隐藏层的维度
        num_layers:编码解码器的层数
        n_head:多头注意力机制中头的个数
        """
        super().__init__()
        # WordEmbedding
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=embed_dim
        )
        # 进行位置编码
        self.positional_encoding = PositionalEncoding(embed_dim, dropout=0.05)
        # 定义transformer模型
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=n_head,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            activation="gelu",  # 前馈神经网络的激活函数
            batch_first=False,  # 输入张量的维度顺序
        )
        # 输出层，全连接神经网络，将transformer输出映射为词汇表，这个位置最可能是哪个词
        self.head = nn.Linear(embed_dim, vocab_size)

    def forward(
        self,
        inp_txt,
        trg_txt,
        src_mask,
        trg_mask,
        src_padding_mask,
        trg_padding_mask,
        memory_key_padding_mask,
    ):
        """
        前向传播:模型的主要计算部分，由一个transformer和一个输出端的全连接神经网络构成，
        最后返回一个vocab_size大小的向量，为各个词在该位置出现的概率
        inp_txt:输入序列（编码器入口）
        trg_txt:目标序列（解码器入口）
        src_mask:输入序列掩码
        trg_mask:目标序列掩码
        src_padding_mask:输入序列padding掩码
        trg_padding_mask:目标序列padding掩码
        memory_key_padding_mask:记忆键填充掩码
        """
        # 输入序列（编码器入口）的wordEmbedding位置编码
        inp_emb = self.positional_encoding(self.embedding(inp_txt))
        # 目标序列（解码器入口）的wordEmbedding位置编码
        trg_emb = self.positional_encoding(self.embedding(trg_txt))
        # 模型部分
        trans_out = self.transformer(
            inp_emb,
            trg_emb,
            src_mask,
            trg_mask,
            None,
            src_padding_mask,
            trg_padding_mask,
            memory_key_padding_mask,
        )
        return self.head(trans_out)

    def encode(self, src, src_mask):
        """
        src:输入序列
        src_mask:输入序列的掩码
        """
        return self.transformer.encoder(
            self.positional_encoding(self.embedding(src)), src_mask
        )

    def decode(self, trg, memory, trg_mask):
        """

        """
        return self.transformer.decoder(
            self.positional_encoding(self.embedding(trg)), memory, trg_mask
        )

    def generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        mask = (torch.triu(torch.ones((sz, sz))) == 1).transpose(0, 1)
        # 转换掩码，把True和False矩阵--->True为0，False为-inf
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask
    def create_mask(
        self, src: torch.Tensor, tgt: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        src:输入序列向量
        tgt:目标序列向量
        """
        # 获取长度
        src_seq_len = src.shape[0]
        tgt_seq_len = tgt.shape[0]
        # 目标输入的上三角掩码
        tgt_mask = self.generate_square_subsequent_mask(tgt_seq_len)
        # 全0(False)矩阵，输入序列不需要掩码
        src_mask = torch.zeros((src_seq_len, src_seq_len)).type(torch.bool)
        # 生成填充掩码，填充的位置为True
        src_padding_mask = (src == config.PAD_ID).transpose(0, 1)
        tgt_padding_mask = (tgt == config.PAD_ID).transpose(0, 1)
        return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask
