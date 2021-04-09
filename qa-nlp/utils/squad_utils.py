import torch
from matplotlib import pyplot as plt
import collections
from itertools import zip_longest
from typing import Callable, List, Tuple


# Define PAD and UNK tokens
PAD = '<PAD>'
UNK = '<UNK>'


# Lambda for computing the mean of a list
mean: Callable[[List[float]], float] = lambda l: sum(l) / len(l)

# Lambda for transforming a list of tuples into a tuple of lists
to_tuple_of_lists: Callable[[List[Tuple]], Tuple[List]] = lambda list_of_tuples: tuple(map(list, zip(*list_of_tuples)))

# Lambda for transforming a tuple of lists into a list of tuples
to_list_of_tuples: Callable[[Tuple[List]], List[Tuple]] = lambda tuple_of_lists: list(zip(*tuple_of_lists))

# Lambda for iterating with batches (if the length of the sequences does not match with the batch size,
# tuples of empty lists are appended)
batch_iteration: Callable[[List[Tuple], int], zip] = lambda data, batch_size: \
    zip_longest(*[iter(data)] * batch_size, fillvalue=([], [], []))


def squad_loss(p_start: torch.FloatTensor, p_end: torch.FloatTensor,
               y_start: torch.LongTensor, y_end: torch.LongTensor) -> torch.FloatTensor:
    bs = p_start.size(0)
    # Retrieve the probability of the correct start and end indexes
    p_true_start = torch.gather(p_start, 1, y_start.unsqueeze(1))  # equivalent to p_start[y_start]
    p_true_end = torch.gather(p_end, 1, y_end.unsqueeze(1))  # equivalent to p_end[y_end]
    # Sum logarithms of start and end probabilities (adding eps to avoid NaN log)
    eps = 1e-7
    sum_vec = torch.log(torch.add(p_true_start, eps)) + torch.log(torch.add(p_true_end, eps))
    # Sum over batch dimension
    return - sum_vec.sum() / bs


def compute_f1(true_answer, predicted_answer):
    common = collections.Counter(true_answer) & collections.Counter(predicted_answer)

    num_same = sum(common.values())

    if num_same == 0:
        return 0

    precision = 1.0 * num_same / len(predicted_answer)
    recall = 1.0 * num_same / len(true_answer)
    f1 = (2 * precision * recall) / (precision + recall)

    return f1


def get_raw_scores(context: Tuple[List[str]],
                   label_start: List[int],
                   label_end: List[int],
                   p_start: List[int],
                   p_end: List[int]):
    exact_scores = []
    f1_scores = []

    for i, c in enumerate(context):
        true_answer = c[label_start[i]:label_end[i] +1 ]
        predicted_answer = c[p_start[i]:p_end[i] +1 ]

        exact_scores.append(int(true_answer == predicted_answer))
        f1_scores.append(compute_f1(true_answer, predicted_answer))

    return exact_scores, f1_scores


def plot_history(history):
    fig1, axes = plt.subplots(nrows=1, ncols=1, figsize=(7.5, 5))
    plt.suptitle('loss', size='xx-large')
    plt.tight_layout(rect=(0, 0.03, 1, 0.))

    axes.plot(history['loss'], label='train_loss')
    axes.plot(history['val_loss'], label='val_loss')
    axes.set_title('loss')
    axes.set(xlabel='# Epochs')
    axes.grid()
    axes.legend()

    fig2, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 5))
    plt.suptitle('scores', size='xx-large')
    plt.tight_layout(rect=(0, 0.03, 1, 0.95))
    axes[0].plot(history['exact_score'], label='train_exact_score')
    axes[0].plot(history['val_exact_score'], label='val_exact_score')
    axes[0].set_title('exact_score')
    axes[0].set(xlabel='# Epochs')
    axes[0].grid()
    axes[0].legend()

    axes[1].plot(history['f1_score'], label='train_f1_score')
    axes[1].plot(history['val_f1_score'], label='val_f1_score')
    axes[1].set_title('f1_score')
    axes[1].set(xlabel='# Epochs')
    axes[1].grid()
    axes[1].legend()

    fig3, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 5))
    plt.suptitle('distances', size='xx-large')
    plt.tight_layout(rect=(0, 0.03, 1, 0.95))
    axes[0].plot(history['distance_end'], label='train_distance_end')
    axes[0].plot(history['val_distance_end'], label='val_distance_end')
    axes[0].set_title('distance_end')
    axes[0].set(xlabel='# Epochs')
    axes[0].grid()
    axes[0].legend()

    axes[1].plot(history['distance_start'], label='train_distance_start')
    axes[1].plot(history['val_distance_start'], label='val_distance_start')
    axes[1].set_title('distance_start')
    axes[1].set(xlabel='# Epochs')
    axes[1].grid()
    axes[1].legend()
