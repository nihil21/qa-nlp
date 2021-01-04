import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class ConvolutionalHighwayNetwork(nn.Module):
    def __init__(self,
                 kernel: Tuple[int] = (5, 5),
                 input_embedding_dim: int = 600,
                 ):
        super(ConvolutionalHighwayNetwork, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1,
                               out_channels=1,
                               kernel_size=kernel,
                               stride=1,
                               padding=(int(kernel[0] / 2), int(kernel[1] / 2)))

        self.gate1 = nn.Conv2d(in_channels=1,
                               out_channels=1,
                               kernel_size=(kernel[0], input_embedding_dim),
                               stride=1,
                               padding=(int(kernel[0] / 2), 0))

        self.conv2 = nn.Conv2d(in_channels=1,
                               out_channels=1,
                               kernel_size=kernel,
                               stride=1,
                               padding=(int(kernel[0] / 2), int(kernel[1] / 2)))

        self.gate2 = nn.Conv2d(in_channels=1,
                               out_channels=1,
                               kernel_size=(kernel[0], input_embedding_dim),
                               stride=1,
                               padding=(int(kernel[0] / 2), 0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # (batch_size, word_length, input_embedding_dim) -> (batch_size, input_channel = 1, word_length, input_embedding_dim):
        x = x.unsqueeze(1)

        h = self.conv1(x)

        t_activation = self.gate1(x)
        t = torch.sigmoid(torch.mean(t_activation, dim=2))
        t.unsqueeze_(3)

        y = F.relu((1 - t) * x + t * h)

        h = self.conv2(y)

        t_activation = self.gate2(y)
        t = torch.sigmoid(torch.mean(t_activation, dim=2))
        t.unsqueeze_(3)

        output = F.relu((1 - t) * y + t * h)
        output.squeeze_(1)

        # (batch_size, word_length, input_embedding_dim)
        return output
