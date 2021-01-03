import torch
import torch.nn as nn
from typing import Optional


class WordEmbedder(nn.Module):
    
    def __init__(self,
                 init_emb: torch.Tensor,
                 trainable: Optional[bool] = False):
        super(WordEmbedder, self).__init__()
        # Create embedding layer
        self.embedding = nn.Embedding(*init_emb.shape)
        if init_emb is not None:  # initialize weights to the embeddings provided
            self.embedding.load_state_dict({'weight': init_emb})
            self.embedding.weight.requires_grad = trainable
        self.word_emb_dim = init_emb.shape[1]

    def forward(self, x: torch.LongTensor) -> torch.FloatTensor:
        return self.embedding(x)
