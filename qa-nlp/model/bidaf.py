import torch
import torch.nn as nn
import torch.nn.functional as F
from model.char_embedder import CharEmbedder
from model.word_embedder import WordEmbedder
from model.convolutional_highway_network import ConvolutionalHighwayNetwork


class BiDAF(nn.Module):

    def __init__(self,
                 char_embedder: CharEmbedder,
                 train_word_embedder: WordEmbedder,
                 eval_word_embedder: WordEmbedder,
                 use_lstm: bool = False,
                 use_constraint: bool = False,
                 use_dropout: bool = False):
        super(BiDAF, self).__init__()

        self.use_constraint = use_constraint

        # Step 1 & 2: Character and Word embedding
        self.char_embedder = char_embedder
        self.train_word_embedder = train_word_embedder
        self.eval_word_embedder = eval_word_embedder
        self.d = char_embedder.char_emb_dim + train_word_embedder.word_emb_dim

        # Highway network to process character + word concatenated embeddings
        self.highway_net = ConvolutionalHighwayNetwork(input_embedding_dim=self.d)

        # Step 3: Contextual embedding layer
        self.ctx_rnn = nn.GRU(input_size=self.d, hidden_size=self.d, bidirectional=True, batch_first=True,
                              dropout=0.2 if use_dropout else 0)  # shared between context and query

        # Step 4: Attention flow
        self.w_s = nn.Linear(in_features=6 * self.d, out_features=1, bias=False)

        # Step 5: Modelling layer
        # bidirectional = True -> concat output
        # num_layers = 2 -> average output
        self.mod_rnn = nn.LSTM(input_size=8 * self.d, hidden_size=self.d, bidirectional=True, num_layers=2,
                               batch_first=True, dropout=0.2 if use_dropout else 0) \
            if use_lstm else nn.GRU(input_size=8 * self.d, hidden_size=self.d, bidirectional=True, num_layers=2,
                                    batch_first=True, dropout=0.2 if use_dropout else 0)  # use LSTM or GRU

        # Step 6: Output layer
        self.w_p_start = nn.Linear(in_features=10 * self.d, out_features=1, bias=False)
        if use_constraint:  # concatenate p_start -> 2d + 1 input size
            self.p_end_rnn = nn.LSTM(input_size=2 * self.d + 1, hidden_size=self.d, bidirectional=True,
                                     batch_first=True, dropout=0.2 if use_dropout else 0) \
                if use_lstm else nn.GRU(input_size=2 * self.d + 1, hidden_size=self.d, bidirectional=True,
                                        batch_first=True, dropout=0.2 if use_dropout else 0)  # use LSTM or GRU
        else:  # standard version -> 2d input size
            self.p_end_rnn = nn.LSTM(input_size=2 * self.d, hidden_size=self.d, bidirectional=True,
                                     batch_first=True, dropout=0.2 if use_dropout else 0) \
                if use_lstm else nn.GRU(input_size=2 * self.d, hidden_size=self.d, bidirectional=True,
                                        batch_first=True, dropout=0.2 if use_dropout else 0)  # use LSTM or GRU

        self.w_p_end = nn.Linear(in_features=10 * self.d, out_features=1, bias=False)

    def _get_contextual_embedding(self, batch_word_seq: torch.LongTensor, batch_char_seq: torch.LongTensor):
        # Step 1 & 2: Word and Character embedding
        char_emb = self.char_embedder(batch_char_seq)
        word_emb = self.train_word_embedder(batch_word_seq) if self.training \
            else self.eval_word_embedder(batch_word_seq)  # word embedding depends on training/eval phase
        # Apply highway network
        merged_emb = self.highway_net(torch.cat([char_emb, word_emb], dim=2))
        # Step 3: Contextual embedding layer
        ctx_emb, _ = self.ctx_rnn(merged_emb)
        return ctx_emb

    def forward(self,
                batch_context_word: torch.LongTensor,
                batch_context_char: torch.LongTensor,
                batch_query_word: torch.LongTensor,
                batch_query_char: torch.LongTensor):
        # Apply step 1 to 3 for context
        ctx_emb_c = self._get_contextual_embedding(batch_context_word, batch_context_char)
        # Apply step 1 to 3 for query
        ctx_emb_q = self._get_contextual_embedding(batch_query_word, batch_query_char)
        # Step 4: Attention flow
        bs, t, _ = ctx_emb_c.shape
        _, j, _ = ctx_emb_q.shape
        # Unsqueeze and expand to make both matrices match shape (bs, t, j, 2d)
        h = ctx_emb_c.unsqueeze(2).expand(bs, t, j, 2 * self.d)  # (bs, t, 2d) -> (bs, t, 1, 2d) -> (bs, t, j, 2d)
        u = ctx_emb_q.unsqueeze(1).expand(bs, t, j, 2 * self.d)  # (bs, j, 2d) -> (bs, 1, j, 2d) -> (bs, t, j, 2d)
        # Compute similarity matrix
        alpha_input = torch.cat([h, u, h * u], dim=-1)  # (bs, t, j, 6d)
        sim_mtx = self.w_s(alpha_input).squeeze(3)  # (bs, t, j, 1) -> (bs, t, j)
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
        m, _ = self.mod_rnn(g)  # (bs, t, 2d)
        # Step 6: Output layer
        p_start = F.softmax(self.w_p_start(torch.cat([g, m], dim=-1)), dim=1)  # (bs, t)
        # If specified, concatenate p_start with RNN input to impose constraint
        m_2, _ = self.p_end_rnn(torch.cat([m, p_start], dim=-1)) \
            if self.use_constraint else self.p_end_rnn(m)  # (bs, t, 2d)
        p_end = F.softmax(self.w_p_end(torch.cat([g, m_2], dim=-1)), dim=1)  # (bs, t)
        return p_start.squeeze(2), p_end.squeeze(2)
