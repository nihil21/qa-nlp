import torch
import torch.nn as nn
import pandas as pd
import numpy as np

from typing import List


class CharacterOneHotEncoder:
    def __init__(self,
                 dataframe: pd.DataFrame,
                 context_list: List[str],
                 encoding_dimension: int = 100):

        self.encoding_dimension = encoding_dimension

        # extracting unique characters from contexts and questions
        unique_chars = {}

        for i, row in dataframe.iterrows():
            for word in row['question'].split(' '):
                for c in word:
                    if c not in unique_chars:
                        unique_chars[c] = 0
                    unique_chars[c] += 1

            for word in context_list[row['context_index']].split(' '):
                for c in word:
                    if c not in unique_chars:
                        unique_chars[c] = 0
                    unique_chars[c] += 1

        self.selected_chars = []

        for element in sorted(((v, k) for k, v in unique_chars.items()), reverse=True):
            # only the 'encoding_dimension' most frequent characters are considered for the one-hot encoding
            # the not selected ones are encoded as unknown (UNK)
            if len(self.selected_chars) < encoding_dimension - 1:
                self.selected_chars.append(element[1])
            else:
                break

        one_hot_chars = np.zeros((len(self.selected_chars) + 1, len(self.selected_chars) + 1))
        np.fill_diagonal(one_hot_chars, 1)

        self.char_to_onehot = {}

        for i in range(len(self.selected_chars)):
            self.char_to_onehot[self.selected_chars[i]] = one_hot_chars[i]

        self.char_to_onehot['UNK'] = one_hot_chars[-1]

    def __get_char_onehot(self,
                          character: str) -> np.array:
        # Given a character, return its one-hot encoding

        assert len(character) == 1, "Error: expected char got string"

        if character in self.selected_chars:
            return self.char_to_onehot[character]
        else:
            return self.char_to_onehot['UNK']

    def get_word_onehot(self,
                        word: str,
                        max_word_len: int) -> np.array:

        # Given a word, return its list of one-hot encoded characters
        # considering padding (as a zeros vector is created)
        word_encoding = np.zeros((max_word_len, self.encoding_dimension))

        for i in range(len(word)):
            word_encoding[i] = self.__get_char_onehot(word[i])

        return word_encoding


class CharacterEmbedder(nn.Module):
    def __init__(self,
                 input_channels: int = 1,
                 output_channels: int = 600,
                 kernel_height: int = 3,
                 input_char_embedding_dimension: int = 100,
                 output_char_embedding_dimension: int = 300,
                 hidden_dim: int = 400):
        super(CharacterEmbedder, self).__init__()

        self.conv_layer = nn.Conv2d(in_channels=input_channels,
                                    out_channels=output_channels,
                                    kernel_size=(kernel_height, input_char_embedding_dimension),
                                    stride=1,
                                    padding=(1, 0))

        self.fc1 = nn.Linear(in_features=output_channels, out_features=hidden_dim)
        self.fc2 = nn.Linear(in_features=hidden_dim, out_features=output_char_embedding_dimension)

    def forward(self, x):
        # x.shape = (batch_size, input_channel = 1, word_length, input_char_embedding_dim)

        x = self.conv_layer(x)
        x = x.squeeze(3)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool1d(x, kernel_size=x.shape[2])
        x = x.squeeze(2)

        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        x = nn.functional.relu(x)

        # x.shape = (batch_size, output_char_embedding_dimension)
        return x
