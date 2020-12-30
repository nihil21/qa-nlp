import torch
import torch.nn as nn
from typing import Optional


class WordEmbedding(nn.Module):
    
    def __init__(self,
                 word_emb_dim: int,
                 word_vocab_dim: int,
                 init_emb: Optional[torch.Tensor] = None):
        super(WordEmbedding, self).__init__()
        # Create embedding layer
        self.embedding = nn.Embedding(word_vocab_dim, word_emb_dim)
        if init_emb is not None:  # initialize weights to the embeddings provided (still trainable)
            self.embedding.load_state_dict({'weight': init_emb})

    def forward(self, x: torch.LongTensor):
        return self.embedding(x)
