import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from char_embedder import CharacterOneHotEncoder
from typing import List, Dict, Tuple, Optional


class Embedder:
    def __init__(self,
                 word_embedding_matrix: np.ndarray,
                 char_one_hot_encoder: CharacterOneHotEncoder,
                 word_to_idx: Dict[str, int]):

        self.word_embedding_matrix = word_embedding_matrix
        self.char_one_hot_encoder = char_one_hot_encoder
        self.word_to_idx = word_to_idx

        self.word_embedding_dim = word_embedding_matrix.shape[1]

    def get_embedding(self,
                      batch_word_seq: Tuple[List[str]]) -> (torch.FloatTensor, torch.LongTensor, torch.IntTensor):

        # Find max length of a sentence (number of words) in batch
        max_sentence_len = max(map(len, batch_word_seq))

        # Find max length of a word (number of chars) in batch
        max_word_len = max([max(map(len, b)) for b in batch_word_seq])

        batch_word_embedding = []
        batch_char_embedding = []
        lengths = []
        for word_seq in batch_word_seq:
            if word_seq:  # Handle partial batches
                orig_len = len(word_seq)
                pad_len = max_sentence_len - orig_len
                # Keep track of original length
                lengths.append(orig_len)

                # Pad sequence
                padded_seq = word_seq + [PAD] * pad_len
                # Embedding sequence of words using word embedding matrix (word encoding) and
                # char one-hot encoding (char encoding)
                embedded_seq = [(self.word_embedding_matrix[self.word_to_idx[w]],
                                self.char_one_hot_encoder.get_word_onehot(w, max_word_len))
                                for w in padded_seq]

                word_embedding, char_embedding = to_tuple_of_lists(embedded_seq)

                batch_word_embedding.append(word_embedding)
                batch_char_embedding.append(char_embedding)

        return torch.cuda.FloatTensor(batch_word_embedding, device=DEVICE),\
               torch.cuda.LongTensor(batch_char_embedding, device=DEVICE),\
               torch.cuda.IntTensor(lengths, device=DEVICE)
