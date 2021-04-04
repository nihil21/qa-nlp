# qa-nlp

## Authors
* [Lorenzo Mario Amorosa](https://github.com/Lostefra)
* [Andrea Espis](https://github.com/97andrea97)
* [Mattia Orlandi](https://github.com/nihil21)
* [Giacomo Pinardi](https://github.com/GiacomoPinardi)

## Summary

Question Answering (QA) task has gained significant popularity over the past years. The implementations
of such systems have varied across the years, starting from knowledge base technologies to deep learning
approaches based on recurrent neural networks, attention mechanisms and transformers.

In particular, the attention mechanism is often implemented as follows:
1. the computed attention weights are used to extract the most relevant information from the context
for answering the question by summarizing it into a fixed-size vector;
2. they are temporally dynamic, since the attention weights of a certain time step are a function of
the attended vector at the previous time step;
3. they are usually uni-directional, namely the query attends on the context.

We have implemented the Bi-Directional Attention Flow (BiDAF) network (Seo et al. 2018): it consists
in a hierarchical multi-stage architecture for modeling the representations of the contexts and the queries
at different levels of granularity. These levels include character-level embedding following Kim 2014,
word-level embedding with GloVe (Pennington, Socher, and Manning 2014) and contextual embedding
using recurrent neural networks.

Notably, it tries to improve some issues of regular attention:
1. the attention layer does not summarize the context into a fixed-size vector, but instead it computes
attention at each time step; then, the attended vector, along with the representations from previous
layers, is allowed to flow to the subsequent layer (called modelling layer );
2. the attention mechanism is memory-less, namely at each time step the attention is a function of
only the context and the query from the current time step (this simplification should lead to the
division of labor between the attention layer, focusing on learning the attention between context
and query, and the modelling layer, focusing on learning the interaction within the query-aware
context representation).
3. the attention flow is both directions, from context to query and vice versa, providing complementary
information to each other.

In this work we have trained our model using the Stanford Question Answering Dataset (SQuAD)
from Rajpurkar et al. 2016.

The models are implemented in PyTorch.
