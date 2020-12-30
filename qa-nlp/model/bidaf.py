import torch
import torch.nn as nn
import torch.nn.functional as F
from model.char_embedder import CharacterEmbedder
from model.word_embedding import WordEmbedding
from model.convolutional_highway_network import HighwayNet
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
        # Step 4: Attention flow
        bs, t, _ = ctx_emb_c.shape
        _, j, _ = ctx_emb_q.shape
        # Unsqueeze and expand to make both matrices match shape (bs, t, j, 2d)
        h = ctx_emb_c.unsqueeze(2).expand(bs, t, j, 2 * self.d)  # (bs, t, 2d) -> (bs, t, 1, 2d) -> (bs, t, j, 2d)
        u = ctx_emb_q.unsqueeze(1).expand(bs, t, j, 2 * self.d)  # (bs, j, 2d) -> (bs, 1, j, 2d) -> (bs, t, j, 2d)
        # Compute similarity matrix
        alpha_input = torch.cat([h, u, h * u], dim=-1)  # (bs, t, j, 6d)
        sim_mtx = self.w_s(alpha_input).squeeze()  # (bs, t, j, 1) -> (bs, t, j)
        # Step 4.1: C2Q
        attention_a = F.softmax(sim_mtx, dim=1)  # (bs, t, j)
        u_tilde = torch.bmm(attention_a, ctx_emb_q)  # (bs, t, j) * (bs, j, 2d) -> (bs, t, 2d)
        # Step 4.2: Q2C
        attention_b = F.softmax(torch.max(sim_mtx, dim=2)[0], dim=1)  # (bs, t)
        h_tilde = torch.bmm(attention_b.unsqueeze(1), ctx_emb_c)  # (bs, 1, t) * (bs, t, 2d) -> (bs, 1, 2d)
        h_tilde = h_tilde.expand((bs, t, 2 * self.d))  # (bs, 1, 2d) -> (bs, t, 2d)
        # Merge C2Q and Q2C
        g = torch.cat([ctx_emb_c, u_tilde, ctx_emb_c * u_tilde, ctx_emb_c * h_tilde], dim=-1)  # (bs, t, 8d)
        # Step 5: Modelling layer
        m = self.mod_rnn(g)  # (bs, t, 2d)
        # Step 6: Output layer
        p_start = F.softmax(self.w_p_start(torch.cat([g, m], dim=-1)), dim=1)  # (bs, t)
        m_2 = self.out_rnn(m)  # (bs, t, 2d)
        p_end = F.softmax(self.w_p_end(torch.cat([g, m_2], dim=-1)), dim=1)  # (bs, t)
        return p_start, p_end

# IDEAS:
# 1. Use NN for g
# 2. Create connection between p_start and p_end
# 3. Add a constraint to take into account that index(max(p_end)) > index(max(p_start)) (in the loss or in point 2)
