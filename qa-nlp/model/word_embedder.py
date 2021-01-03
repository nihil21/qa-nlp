import torch
import torch.nn as nn
from typing import Optional


class WordEmbedder(nn.Module):
    
    def __init__(self,
                 word_emb_dim: int,
                 word_vocab_dim: int,
                 init_emb: Optional[torch.Tensor] = None,
                 trainable: Optional[bool] = False):
        super(WordEmbedder, self).__init__()
        # Create embedding layer
        self.embedding = nn.Embedding(word_vocab_dim, word_emb_dim)
        if init_emb is not None:  # initialize weights to the embeddings provided
            self.embedding.load_state_dict({'weight': init_emb})
            self.embedding.weight.requires_grad = trainable
        self.word_emb_dim = word_emb_dim

    def forward(self, x: torch.LongTensor):
        return self.embedding(x)
