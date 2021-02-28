import torch.nn as nn
import torch.nn.functional as F
from transformers import AlbertModel


class AlbertForQuestionAnswering(nn.Module):
    def __init__(self, model_name):
        super(AlbertForQuestionAnswering, self).__init__()

        self.albert = AlbertModel.from_pretrained(model_name)

        # Freeze Albert parameters
        for param in self.albert.parameters():
            param.requires_grad = False

        self.fc = nn.Linear(self.albert.config.hidden_size, 2)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        sequence_output = self.albert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)[0]

        logits = self.fc(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)

        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        start_pred = nn.functional.softmax(start_logits, dim=-1)
        end_pred = nn.functional.softmax(end_logits, dim=-1)

        return start_pred, end_pred
