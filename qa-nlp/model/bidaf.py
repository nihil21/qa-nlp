import torch
import torch.nn as nn
import torch.nn.functional as F
from model.char_embedding import CharEmbedding
from model.word_embedding import WordEmbedding
from model.highway_net import HighwayNet
from typing import List, Tuple, Optional


class BiDAF(nn.Module):

    def __init__(self,
                 char_emb_dim: int,
                 char_vocab_dim: int,
                 char_out_dim: int,
                 kernels: List[Tuple[int]],
                 word_emb_dim: int,
                 word_vocab_dim: int,
                 use_dropout: Optional[bool] = False):
        super(BiDAF, self).__init__()
        self.d = char_emb_dim + word_emb_dim
        # Step 1: Character embedding
        self.char_embedding = CharEmbedding(char_emb_dim, char_vocab_dim, char_out_dim, kernels, use_dropout)
        # Step 2: Word embedding
        self.char_embedding = WordEmbedding(word_emb_dim, word_vocab_dim)
        # Highway network to process character + word concatenated embeddings
        self.highway_net = HighwayNet()
        # Step 3: Contextual embedding layer
        self.ctx_rnn = nn.GRU(input_size=self.d, hidden_size=self.d, bidirectional=True, batch_first=True,
                              dropout=0.2 if use_dropout else 0)  # shared between context and query
        # Step 4: Attention flow
        self.w_s = nn.Linear(in_features=6 * self.d, out_features=1, bias=False)
        # Step 5: Modelling layer
        # bidirectional = True -> concat output
        # num_layers = 2 -> average output
        self.mod_rnn = nn.GRU(input_size=8 * self.d, hidden_size=self.d, bidirectional=True, num_layers=2,
                              batch_first=True, dropout=0.2 if use_dropout else 0)
        # Step 6: Output layer
        self.w_p_start = nn.Linear(in_features=10 * self.d, out_features=1, bias=False)
        self.out_rnn = nn.GRU(input_size=2 * self.d, hidden_size=self.d, bidirectional=True, batch_first=True,
                              dropout=0.2 if use_dropout else 0)
        self.w_p_end = nn.Linear(in_features=10 * self.d, out_features=1, bias=False)

    def _get_contextual_embedding(self, char_tensor: torch.LongTensor, word_tensor: torch.LongTensor):
        # Step 1: Character embedding
        char_emb = self.char_embedding(char_tensor)
        # Step 2: Word embedding
        word_emb = self.word_embedding(word_tensor)
        # Apply highway network
        merged_emb = self.highway_net(torch.cat([char_emb, word_emb], dim=2))
        # Step 3: Contextual embedding layer
        ctx_emb, _ = self.ctx_rnn(merged_emb)
        return ctx_emb

    def forward(self,
                word_c: torch.LongTensor,
                char_c: torch.LongTensor,
                word_q: torch.LongTensor,
                char_q: torch.LongTensor):
        # Apply step 1 to 3 for context
        ctx_emb_c = self._get_contextual_embedding(char_c, word_c)
        # Apply step 1 to 3 for query
        ctx_emb_q = self._get_contextual_embedding(char_q, word_q)































"""
class BiDAF(nn.Module):
    
    def __init__(self, w_embd_size):
        super().__init__()
        self.embd_size = w_embd_size
        self.d = self.embd_size * 2  # word_embedding + char_embedding
        # self.d = self.embd_size # only word_embedding

        self.char_embd_net = CharEmbedding(args)
        self.word_embd_net = WordEmbedding(args)
        self.highway_net = Highway(self.d)
        self.ctx_embd_layer = nn.GRU(self.d, self.d, bidirectional=True, dropout=0.2, batch_first=True)

        self.W = nn.Linear(6*self.d, 1, bias=False)

        self.modeling_layer = nn.GRU(8*self.d, self.d, num_layers=2, bidirectional=True, dropout=0.2, batch_first=True)

        self.p1_layer = nn.Linear(10*self.d, 1, bias=False)
        self.p2_lstm_layer = nn.GRU(2*self.d, self.d, bidirectional=True, dropout=0.2, batch_first=True)
        self.p2_layer = nn.Linear(10*self.d, 1)

    def forward(self):
        batch_size = ctx_w.size(0)
        T = ctx_w.size(1)    # context sentence length (word level)
        J = query_w.size(1)  # query sentence length (word level)

        # 1. Character Embedding Layer
        # 2. Word Embedding Layer
        # 3. Contextual  Embedding Layer
        embd_context = self.build_contextual_embd(ctx_c, ctx_w)    # (N, T, 2d)
        embd_query = self.build_contextual_embd(query_c, query_w)  # (N, J, 2d)

        # 4. Attention Flow Layer
        # Make a similarity matrix
        shape = (batch_size, T, J, 2 * self.d)           # (N, T, J, 2d)
        embd_context_ex = embd_context.unsqueeze(2)      # (N, T, 2d) -> (N, T, 1, 2d)
        embd_context_ex = embd_context_ex.expand(shape)  # (N, T, 1, 2d) -> (N, T, J, 2d)
        embd_query_ex = embd_query.unsqueeze(1)          # (N, J, 2d) -> (N, 1, J, 2d)
        embd_query_ex = embd_query_ex.expand(shape)      # (N, 1, J, 2d) -> (N, T, J, 2d)
        a_elmwise_mul_b = torch.mul(embd_context_ex, embd_query_ex)  # (N, T, J, 2d)
        cat_data = torch.cat((embd_context_ex, embd_query_ex, a_elmwise_mul_b), 3)  # (N, T, J, 6d), [h;u;h◦u] (2d + 2d + 2d)
        S = self.W(cat_data).view(batch_size, T, J)  # (N, T, J)

        # Context2Query
        c2q = torch.bmm(F.softmax(S, dim=-1), embd_query) # (N, T, 2d) = bmm( (N, T, J), (N, J, 2d) )
        # Query2Context
        # b: attention weights on the context
        b = F.softmax(torch.max(S, 2)[0], dim=-1)  # (N, T)
        q2c = torch.bmm(b.unsqueeze(1), embd_context)  # (N, 1, 2d) = bmm( (N, 1, T), (N, T, 2d) )
        q2c = q2c.repeat(1, T, 1)  # (N, T, 2d), tiled T times

        # G: query aware representation of each context word
        G = torch.cat((embd_context, c2q, embd_context.mul(c2q), embd_context.mul(q2c)), 2)  # (N, T, 8d)

        # 5. Modeling Layer
        M, _h = self.modeling_layer(G)  # M: (N, T, 2d)

        # 6. Output Layer
        G_M = torch.cat((G, M), 2)  # (N, T, 10d)
        p1 = F.softmax(self.p1_layer(G_M).squeeze(), dim=-1)  # (N, T)

        M2, _ = self.p2_lstm_layer(M)  # (N, T, 2d)
        G_M2 = torch.cat((G, M2), 2)  # (N, T, 10d)
        p2 = F.softmax(self.p2_layer(G_M2).squeeze(), dim=-1)  # (N, T)

        return p1, p2
"""
