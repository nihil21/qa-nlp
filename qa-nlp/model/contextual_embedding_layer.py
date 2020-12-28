import torch
import torch.nn as nn

class ContextualEmbeddingLayer (nn.Module):
    def __init__ (self,
                  input_embedding_dim: int = 600,
                  ):
        super(ContextualEmbeddingLayer, self).__init__()

        self.lstm = nn.LSTM(input_size=input_embedding_dim,
                            hidden_size=input_embedding_dim,
                            bidirectional=True,
                            batch_first=True)

    def forward (self, x):
        # x.shape = (batch_size, word_length, embedding_dim)

        output, _ = self.lstm(x)

        return output
