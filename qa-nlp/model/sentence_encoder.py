from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from typing import Optional


class SentenceEncoder(ABC, nn.Module):
    @abstractmethod
    def get_output_dim(self):
        pass

    @abstractmethod
    def forward(self, x: torch.Tensor, x_lengths: Optional[torch.IntTensor] = None) -> torch.Tensor:
        pass
