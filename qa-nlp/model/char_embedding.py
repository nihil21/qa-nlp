import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional


class CharEmbedding(nn.Module):
    
    def __init__(self,
                 char_emb_dim: int,
                 char_vocab_dim: int,
                 char_out_dim: int,
                 kernels: List[Tuple[int]],
                 use_dropout: Optional[bool] = False):
        super(CharEmbedding, self).__init__()
        self.emb_dim = char_emb_dim
        # Create embedding layer
        self.embedding = nn.Embedding(char_vocab_dim, char_emb_dim)
        # Create list of convolutions
        self.conv = nn.ModuleList([nn.Conv2d(1, char_out_dim, (k[0], k[1])) for k in kernels])
        # Apply dropout for regularization, if required
        self.dropout = nn.Dropout(0.2) if use_dropout else nn.Module()

    def forward(self, x: torch.LongTensor):  # x contains indexes of characters
        input_shape = x.size()  # (batch_size, seq_len, word_len)

        # h e  l  l  o
        # 7 4 12 12 17
        raise NotImplementedError("TODO")
