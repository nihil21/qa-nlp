#!/usr/bin/env python

# System modules
import json
import os
import sys
import pickle
from itertools import zip_longest

# Deep learning framework
import torch

# Natural language tools
import nltk
from nltk.tokenize import TreebankWordTokenizer

# Import custom modules
from model.bidaf import BiDAF
from model.tensor_maker import TensorMaker

# Other tools
import pandas as pd
from tqdm import tqdm

# Type hint
import typing

# Set device to GPU if available
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Lambda for transforming a list of tuples into a tuple of lists
to_tuple_of_lists: typing.Callable[[typing.List[typing.Tuple]], typing.Tuple[typing.List]] = \
    lambda list_of_tuples: tuple(map(list, zip(*list_of_tuples)))

# Lambda for iterating with batches
# (if the length of the sequences does not match with the batch size, tuples of empty lists are appended)
batch_iteration: typing.Callable[[typing.List[typing.Tuple]], zip] = lambda data, batch_size: \
    zip_longest(*[iter(data)] * batch_size, fillvalue=([], [], []))


def tokenize_corpus(df: pd.DataFrame, context_list: typing.List[str]):
    twt = TreebankWordTokenizer()
    
    # Retrieve contexts
    contexts = df['context_index'].apply(lambda x: context_list[x])
    # Tokenize both contexts and queries
    ctx = contexts.apply(lambda x: twt.tokenize(x)).tolist()
    qry = df['question'].apply(lambda x: twt.tokenize(x)).tolist()
    
    # Get spans of tokens, to revert the tokenization
    spans_list = contexts.apply(lambda x: list(twt.span_tokenize(x))).tolist()
    
    return ctx, qry, spans_list


def generate_evaluation_json(model: BiDAF,
                             tensor_maker: TensorMaker,
                             evaluation_data: typing.List[typing.Tuple[str, typing.List[str], typing.List[str]]],
                             spans_list: typing.List[typing.List[typing.Tuple[int, int]]],
                             id_list: typing.List[str],
                             filename: str):
    predictions = {}
 
    with torch.no_grad():
        # Create batch iterator with size = 1
        batch_iter = batch_iteration(evaluation_data, batch_size=1)
 
        for i, batch in enumerate(tqdm(batch_iter, total=len(evaluation_data), leave=True)):
            # Extract samples
            batch_context, batch_context_tokenized, batch_query_tokenized = to_tuple_of_lists(batch)

            context_word_tensor, context_char_tensor, _ = tensor_maker.get_tensor(batch_context_tokenized)
            query_word_tensor, query_char_tensor, _ = tensor_maker.get_tensor(batch_query_tokenized)

            # Make prediction
            p_soft_start, p_soft_end = model(context_word_tensor, context_char_tensor,
                                             query_word_tensor, query_char_tensor)

            # Argmax
            p_start = torch.argmax(p_soft_start, dim=1)[0]
            p_end = torch.argmax(p_soft_end, dim=1)[0]

            start_word_idx = p_start.item()
            end_word_idx = p_end.item()

            span = spans_list[i]
            start_char_idx = span[start_word_idx][0]
            end_char_idx = span[end_word_idx][1]

            answer = batch_context[0][start_char_idx:end_char_idx + 1]

            idx = id_list[i]
            predictions[idx] = answer
 
    with open(filename, "w") as f:
        f.write(json.dumps(predictions))


def main():
    if len(sys.argv) != 2:
        sys.exit('Usage: python3 compute_answers.py dataset.json')
    filename = sys.argv[1]

    # Download NLTK components
    nltk.download('punkt')
    nltk.download('stopwords')

    # Use GPU acceleration if possible
    print('Using this device:', DEVICE)
    
    # To avoid memory problems disable cuDNN
    torch.backends.cudnn.enabled = False

    # Read input file and parse data
    with open(filename, 'r') as f:
        raw_data = f.readlines()[0]
    parsed_data = json.loads(raw_data)['data']

    # Create DataFrame object
    context_list = []
    context_index = -1
    paragraph_index = -1

    dataset = {'paragraph_index': [], 'context_index': [], 'question': [], 'id': []}
    for i in range(len(parsed_data)):
        paragraph_index += 1
        for j in range(len(parsed_data[i]['paragraphs'])):
            context_list.append(parsed_data[i]['paragraphs'][j]['context'])
            context_index += 1

            for k in range(len(parsed_data[i]['paragraphs'][j]['qas'])):
                question = parsed_data[i]['paragraphs'][j]['qas'][k]['question']
                idx = parsed_data[i]['paragraphs'][j]['qas'][k]['id']

                dataset['paragraph_index'].append(paragraph_index)
                dataset['context_index'].append(context_index)
                dataset['question'].append(question)
                dataset['id'].append(idx)

    df = pd.DataFrame.from_dict(dataset)
    id_list = df['id'].tolist()

    print('Tokenizing corpus...')

    # Tokenize corpus
    context_tokenized, query_tokenized, spans_list = tokenize_corpus(df, context_list)
    
    print('Done.')
    print('Loading model...')
    
    # Load BiDAF
    model = torch.load(os.path.join('best_model', 'bidaf.pt'))
    # Load tensor maker
    with open(os.path.join('best_model', 'tensor_maker.pickle'), 'rb') as f:
        tensor_maker = pickle.load(f)
    
    print('Done.')
    print('Generating answers...')

    # Retrieve original contexts
    contexts = df['context_index'].apply(lambda x: context_list[x])

    evaluation_data = []
    for i in range(len(context_tokenized)):
        evaluation_data.append((contexts[i], context_tokenized[i], query_tokenized[i]))

    # Generate answers
    generate_evaluation_json(model, tensor_maker, evaluation_data, spans_list, id_list, "predictions.json")
    
    print('Success!')


if __name__ == "__main__":
    main()
