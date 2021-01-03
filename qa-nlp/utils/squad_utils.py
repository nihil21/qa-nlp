import torch
from itertools import zip_longest
from typing import Callable, List, Tuple


# Use GPU acceleration if possible
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("using this device:", DEVICE)

# Define PAD token
PAD = '<PAD>'

# Lambda for computing the mean of a list
mean: Callable[[List[float]], float] = lambda l: sum(l) / len(l)

# Lambda for transforming a list of tuples into a tuple of lists
to_tuple_of_lists: Callable[[List[Tuple]], Tuple[List]] = lambda list_of_tuples: tuple(map(list, zip(*list_of_tuples)))

# Lambda for transforming a tuple of lists into a list of tuples
to_list_of_tuples: Callable[[Tuple[List]], List[Tuple]] = lambda tuple_of_lists: list(zip(*tuple_of_lists))

# Lambda for iterating with batches (if the length of the sequences does not match with the batch size,
# tuples of empty lists are appended)
batch_iteration: Callable[[List[Tuple]], zip] = lambda data, batch_size: \
    zip_longest(*[iter(data)] * batch_size, fillvalue=([], [], ''))
