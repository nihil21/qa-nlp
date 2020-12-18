from sentence_encoder import SentenceEncoder
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import Optional


class QAModel(nn.Module):
    def __init__(self,
                 encoder: SentenceEncoder,
                 classifier: nn.Module,
                 merge_strategy: Optional[str] = 'concat',
                 use_cos_sim: Optional[bool] = False):
        super(QAModel, self).__init__()

        self.encoder = encoder
        self.classifier = classifier
        self.merge_strategy = merge_strategy
        self.use_cos_sim = use_cos_sim

    def forward(self,
                x1: torch.Tensor,
                x2: torch.Tensor,
                x1_lengths: Optional[torch.IntTensor] = None,
                x2_lengths: Optional[torch.IntTensor] = None):
        # Encode both inputs
        x1 = self.encoder(x1, x1_lengths)
        x2 = self.encoder(x2, x2_lengths)
        # Merge according to strategy
        encoding = None
        if self.merge_strategy == 'sum':
            encoding = x1 + x2  # SUM
        elif self.merge_strategy == 'avg':
            encoding = (x1 + x2) / 2  # AVERAGE
        elif self.merge_strategy == 'concat':
            encoding = torch.cat((x1, x2), dim=1)  # CONCATENATION
        # Optionally add cosine similarity feature
        if self.use_cos_sim:
            # Compute cosine similarity between corresponding embeddings inside the batch: the slices are reshaped
            # to (1, batch) to match cosine_similarity API specifications for "single sample" case
            cos_sim = [cosine_similarity(slice_x1.view(1, -1),
                                         slice_x2.view(1, -1)) for slice_x1, slice_x2 in zip(x1.detach().cpu(),
                                                                                             x2.detach().cpu())]
            # Convert the list into torch tensor
            cos_sim = torch.from_numpy(np.concatenate(cos_sim, axis=0)).to(DEVICE)
            # Concatenate cosine similarity to original tensor:
            # encoding has shape (batch, emb_size), cos_sim has shape (batch, 1)
            # torch.cat requires shapes (emb_size, batch) and (1, batch), so both are transposed
            encoding = torch.cat((encoding.transpose(0, 1), cos_sim.transpose(0, 1)), dim=0).transpose(0, 1)

        out = self.classifier(encoding)

        return out
