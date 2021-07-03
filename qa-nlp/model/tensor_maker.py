import torch
import numpy as np
from typing import Dict
from utils.squad_utils import UNK
import typing


class TensorMaker:
    def __init__(self,
                 word_to_idx: Dict[str, int],
                 char_to_idx: Dict[str, int],
                 device: torch.device):
        self.word_to_idx = word_to_idx
        self.char_to_idx = char_to_idx
        self.device = device

    def get_tensor(self, sentences: typing.Sequence[typing.List[str]]) \
            -> (torch.LongTensor, torch.LongTensor, torch.IntTensor):
        # Find max length of a sentence (number of words) in sentences
        max_sentence_len = max(map(len, sentences))
        # Find max length of a word (number of chars) in sentences
        max_word_len = max([max(map(len, sentence)) for sentence in sentences if len(sentence) > 0])
        # Prepare word and char tensors
        word_tensor = np.zeros((len(sentences), max_sentence_len))
        char_tensor = np.zeros((len(sentences), max_sentence_len, max_word_len))

        lengths = []
        for i in range(len(sentences)):  # loop over sentences
            if len(sentences[i]) > 0:  # Handle partial batches
                # Keep track of original length
                lengths.append(len(sentences[i]))
                for j in range(len(sentences[i])):  # loop over words in current sentence
                    word_tensor[i, j] = self.word_to_idx[sentences[i][j]]
                    for k in range(len(sentences[i][j])):  # loop over chars in current word
                        char_tensor[i, j, k] = self.char_to_idx.get(sentences[i][j][k], self.char_to_idx[UNK])

        return torch.cuda.LongTensor(word_tensor, device=self.device), \
            torch.cuda.LongTensor(char_tensor, device=self.device), \
            torch.IntTensor(lengths)  # SUPERFLUO (forse)
