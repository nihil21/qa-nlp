import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class CharEmbedder(nn.Module):

    def __init__(self,
                 init_emb: torch.Tensor,
                 out_char_emb_dim: Optional[int] = 50,
                 hidden_dim: Optional[int] = 64,
                 input_channels: Optional[int] = 1,
                 output_channels: Optional[int] = 100,
                 kernel_height: Optional[int] = 5,
                 trainable: Optional[bool] = False):
        super(CharEmbedder, self).__init__()

        # Create embedding layer (one extra row for padding)
        # Create embedding layer
        self.embedding = nn.Embedding.from_pretrained(init_emb, freeze=not(trainable))
        self.word_emb_dim = init_emb.shape[1]

        input_char_embedding_dimension = init_emb.shape[1]
        self.conv_layer = nn.Conv2d(in_channels=input_channels,
                                    out_channels=output_channels,
                                    kernel_size=(kernel_height, input_char_embedding_dimension),
                                    stride=1,
                                    padding=(2, 0),
                                    bias=False)

        self.fc1 = nn.Linear(in_features=output_channels, out_features=hidden_dim, bias=False)
        self.fc2 = nn.Linear(in_features=hidden_dim, out_features=out_char_emb_dim, bias=False)

        self.char_emb_dim = out_char_emb_dim

    def forward(self, x: torch.LongTensor) -> torch.FloatTensor:
        # (batch_size, seq_len, word_len)
        bs = x.shape[0]
        x = self.embedding(x)  # (batch_size, seq_len, word_len, in_char_emb_dim)
        # (batch_size * seq_len, word_len, input_char_embedding_dim)
        x = x.view(x.shape[0] * x.shape[1], x.shape[2], x.shape[3])
        x = x.unsqueeze(1)  # (batch_size, input_channel = 1, word_len, in_char_emb_dim)
        
        x = self.conv_layer(x)
        x = x.squeeze(3)
        x = F.relu(x)
        x = F.max_pool1d(x, kernel_size=x.shape[2])
        x = x.squeeze(2)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)

        # (batch_size, out_char_emb_dim)
        x = x.unsqueeze(1)  # (batch_size, 1, out_char_emb_dim)
        return x.view(bs, -1, x.shape[2])  # (batch_size, seq_len, out_char_emb_dim)
