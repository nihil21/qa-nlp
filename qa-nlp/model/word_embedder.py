import torch
import torch.nn as nn
from typing import Optional


class WordEmbedder(nn.Module):
    
    def __init__(self,
                 init_emb: torch.Tensor,
                 trainable: Optional[bool] = False):
        super(WordEmbedder, self).__init__()
        # Create embedding layer
        self.embedding = nn.Embedding.from_pretrained(init_emb, freeze = not(trainable))
        self.word_emb_dim = init_emb.shape[1]

    def forward(self, x: torch.LongTensor) -> torch.FloatTensor:
        #print("before embedding:\n", x.shape)
        #print("after embedding:\n", self.embedding(x).shape)
        return self.embedding(x)
