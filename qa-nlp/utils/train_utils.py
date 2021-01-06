import torch
from model.bidaf import BiDAF
from model.tensor_maker import TensorMaker
import random
import numpy as np
from time import time
from tqdm.notebook import tqdm
from itertools import zip_longest
from typing import Callable, List, Tuple, Dict, Optional

# Lambda for computing the mean of a list
mean: Callable[[List[float]], float] = lambda l: sum(l) / len(l)

# Lambda for transforming a list of tuples into a tuple of lists
to_tuple_of_lists: Callable[[List[Tuple]], Tuple[List]] = lambda list_of_tuples: tuple(map(list, zip(*list_of_tuples)))

# Lambda for transforming a tuple of lists into a list of tuples
to_list_of_tuples: Callable[[Tuple[List]], List[Tuple]] = lambda tuple_of_lists: list(zip(*tuple_of_lists))

# Lambda for iterating with batches (if the length of the sequences does not match with the batch size,
# tuples of empty lists are appended)
batch_iteration: Callable[[List[Tuple]], zip] = lambda data, batch_size: \
    zip_longest(*[iter(data)] * batch_size, fillvalue=([], [], []))


# Train function util
def train(model: BiDAF,
          data: List[Tuple[List[str], List[str], Tuple[int, int]]],
          batch_size: int,
          criterion: Callable[[torch.FloatTensor, torch.FloatTensor, torch.LongTensor, torch.LongTensor],
                              torch.FloatTensor],
          optimizer: torch.optim,
          tensor_maker: TensorMaker,
          verbose: Optional[bool] = False,
          scaler: Optional[torch.cuda.amp.grad_scaler.GradScaler] = False) -> (float, int, int):
    loss_data = []
    distance_start = 0
    distance_end = 0
    total = 0
    flag = 0

    # Create batch iterator
    batch_iter = batch_iteration(data, batch_size)
    steps = len(data) // batch_size if len(data) % batch_size == 0 else len(data) // batch_size + 1
    if verbose:
        batch_iter = tqdm(batch_iter, total=steps, leave=False)
    for batch in batch_iter:
        # Extract samples
        batch_context, batch_query, batch_label = to_tuple_of_lists(batch)

        # Filter valid samples in batches (in case of incomplete ones)
        batch_context: Tuple[List[str]] = tuple([c for c in batch_context if len(c) > 0])
        batch_query: Tuple[List[str]] = tuple([q for q in batch_query if len(q) > 0])
        batch_label = [lab for lab in batch_label if len(lab) > 0]

        # Extract start and end indexes
        labels_start, labels_end = to_tuple_of_lists(batch_label)

        total_batch = len(batch_context)

        context_word_tensor, context_char_tensor, context_lengths = tensor_maker.get_tensor(batch_context)
        query_word_tensor, query_char_tensor, query_lengths = tensor_maker.get_tensor(batch_query)
        labels_start = torch.cuda.LongTensor(labels_start)
        labels_end = torch.cuda.LongTensor(labels_end)

        if scaler:
            # Make prediction
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():  # https://pytorch.org/docs/stable/notes/amp_examples.html
                p_soft_start, p_soft_end = model(context_word_tensor, context_char_tensor,
                                                 query_word_tensor, query_char_tensor)
                loss = criterion(p_soft_start, p_soft_end, labels_start, labels_end)
            # Backpropagation
            scaler.scale(loss).backward()  # https://pytorch.org/docs/stable/notes/amp_examples.html
            scaler.step(optimizer)  # https://pytorch.org/docs/stable/notes/amp_examples.html
            scaler.update()  # https://pytorch.org/docs/stable/notes/amp_examples.html
        else:
            # Make prediction
            optimizer.zero_grad()
            p_soft_start, p_soft_end = model(context_word_tensor, context_char_tensor,
                                             query_word_tensor, query_char_tensor)
            loss = criterion(p_soft_start, p_soft_end, labels_start, labels_end)
            # Backpropagation
            loss.backward()
            optimizer.step()

        # Compute distance metric
        p_start = torch.argmax(p_soft_start, dim=1)
        p_end = torch.argmax(p_soft_end, dim=1)
        start_dist = torch.abs(p_start - labels_start).sum()
        end_dist = torch.abs(p_end - labels_end).sum()

        # Update history
        loss_data.append(loss.item())
        distance_start += start_dist.item()
        distance_end += end_dist.item()
        total += total_batch

        if flag == 0:
            print(f'Start (p): {p_start[0].item()}, End (p): {p_end[0].item()}, '
                  f'Start (T): {labels_start[0].item()}, End (T): {labels_end[0].item()}')
            flag += 1

    return mean(loss_data), distance_start // total, distance_end // total


# Evaluate function util
def evaluate(model: BiDAF,
             data: List[Tuple[List[str], List[str], Tuple[int, int]]],
             batch_size: int,
             criterion: Callable[[torch.FloatTensor, torch.FloatTensor, torch.LongTensor, torch.LongTensor],
                                 torch.FloatTensor],
             tensor_maker: TensorMaker,
             verbose: Optional[bool] = False) -> (float, int, int):
    loss_data = []
    distance_start = 0
    distance_end = 0
    total = 0
    flag = 0

    with torch.no_grad():
        # Create batch iterator
        batch_iter = batch_iteration(data, batch_size)
        steps = len(data) // batch_size if len(data) % batch_size == 0 else len(data) // batch_size + 1
        if verbose:
            batch_iter = tqdm(batch_iter, total=steps, leave=False)
        for batch in batch_iter:
            # Extract samples
            batch_context, batch_query, batch_label = to_tuple_of_lists(batch)

            # Filter valid samples in batches (in case of incomplete ones)
            batch_context: Tuple[List[str]] = tuple([c for c in batch_context if len(c) > 0])
            batch_query: Tuple[List[str]] = tuple([q for q in batch_query if len(q) > 0])
            batch_label = [lab for lab in batch_label if len(lab) > 0]

            # Extract start and end indexes
            labels_start, labels_end = to_tuple_of_lists(batch_label)

            total_batch = len(batch_context)

            context_word_tensor, context_char_tensor, context_lengths = tensor_maker.get_tensor(batch_context)
            query_word_tensor, query_char_tensor, query_lengths = tensor_maker.get_tensor(batch_query)
            labels_start = torch.cuda.LongTensor(labels_start)
            labels_end = torch.cuda.LongTensor(labels_end)

            # Make prediction
            p_soft_start, p_soft_end = model(context_word_tensor, context_char_tensor,
                                             query_word_tensor, query_char_tensor)
            # Compute loss
            loss = criterion(p_soft_start, p_soft_end, labels_start, labels_end)

            # Compute distance metric
            p_start = torch.argmax(p_soft_start, dim=1)
            p_end = torch.argmax(p_soft_end, dim=1)
            start_dist = torch.abs(p_start - labels_start).sum()
            end_dist = torch.abs(p_end - labels_end).sum()

            # Update history
            loss_data.append(loss.item())
            distance_start += start_dist.item()
            distance_end += end_dist.item()
            total += total_batch

            if flag == 0:
                print(f'Start (p): {p_start[0].item()}, End (p): {p_end[0].item()}, '
                      f'Start (T): {labels_start[0].item()}, End (T): {labels_end[0].item()}')
                flag += 1

    return mean(loss_data), distance_start // total, distance_end // total


# Training loop function util
def training_loop(model: BiDAF,
                  train_data: List[Tuple[List[str], List[str], Tuple[int, int]]],
                  optimizer: torch.optim,
                  epochs: int,
                  batch_size: int,
                  criterion: Callable[[torch.FloatTensor, torch.FloatTensor, torch.LongTensor, torch.LongTensor],
                                      torch.FloatTensor],
                  train_tensor_maker: TensorMaker,
                  val_tensor_maker: Optional[TensorMaker] = None,
                  lr_scheduler: torch.optim.lr_scheduler = None,
                  val_data: Optional[List[Tuple[List[str], List[str], Tuple[int, int]]]] = None,
                  early_stopping: Optional[bool] = False,
                  patience: Optional[int] = 5,
                  tolerance: Optional[float] = 1e-4,
                  checkpoint_path: Optional[str] = None,
                  verbose: Optional[bool] = True,
                  seed: Optional[int] = 42,
                  mix_scale: Optional[bool] = False) -> (Dict[str, List[float]]):
    # Set seed for reproducibility
    if seed:
        random.seed(seed)

    history = {'loss': [],
               'distance_start': [],
               'distance_end': [],
               'val_loss': [],
               'val_distance_start': [],
               'val_distance_end': []}

    # Initialize variables for early stopping
    min_val_loss = np.inf
    no_improve_counter = 0

    # https://pytorch.org/docs/stable/notes/amp_examples.html
    scaler = torch.cuda.amp.GradScaler() if mix_scale else False

    for ep in range(epochs):
        if verbose:
            print('-' * 100)
            print(f'Epoch {ep + 1}/{epochs}')

        # Shuffle training set at each epoch
        random.shuffle(train_data)

        start = time()
        train_loss, train_distance_start, train_distance_end = train(model, train_data, batch_size, criterion,
                                                                     optimizer, train_tensor_maker, verbose, scaler)
        end = time()

        history['loss'].append(train_loss)
        history['distance_start'].append(train_distance_start)
        history['distance_end'].append(train_distance_end)
        if verbose:
            print(f'\tLoss: {train_loss:.5f} - Distance start: {train_distance_start:d} - '
                  f'Distance end: {train_distance_end:d} [Time elapsed: {end - start:.2f} s]')

        # Do validation if required
        if val_data and val_tensor_maker:
            # Activate eval mode
            model.eval()

            # Shuffle validation set at each epoch
            random.shuffle(val_data)

            start = time()
            val_loss, val_distance_start, val_distance_end = evaluate(model, val_data, batch_size, criterion,
                                                                      val_tensor_maker, verbose)
            end = time()

            history['val_loss'].append(val_loss)
            history['val_distance_start'].append(val_distance_start)
            history['val_distance_end'].append(val_distance_end)
            if verbose:
                print(f'\tValidation loss: {val_loss:.5f} - Distance start: {val_distance_start:.2f} - '
                      f'Distance end: {val_distance_end:.2f} [Time elapsed: {end - start:.2f} s]')

            # Deactivate eval mode
            model.train()

            if early_stopping and checkpoint_path:
                # If validation loss is lower than minimum, update minimum
                if val_loss < min_val_loss - tolerance:
                    min_val_loss = val_loss
                    no_improve_counter = 0

                    # Save model
                    torch.save(model.state_dict(), checkpoint_path)
                # otherwise increment counter
                else:
                    no_improve_counter += 1
                # If loss did not improve for 'patience' epochs, break
                if no_improve_counter == patience:
                    if verbose:
                        print(f'Early stopping: no improvement in validation loss for '
                              f'{patience} epochs from {min_val_loss:.5f}')
                    # Restore model to best
                    model.load_state_dict(torch.load(checkpoint_path))
                    model.eval()
                    break

        # If lr scheduling is used, invoke next step
        if lr_scheduler:
            lr_scheduler.step()

    return history