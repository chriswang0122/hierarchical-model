import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class RNN(nn.Module):

    def __init__(self, vocab_size, embed_size, time_step, hidden_size,
                 dropout_rate, bos_idx, eos_idx, pad_idx):
        super(RNN, self).__init__()
        self.time_step = time_step
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx
        self.pad_idx = pad_idx

        # input
        self.embedding = nn.Embedding(vocab_size, embed_size,
                                      padding_idx=pad_idx)
        # RNN
        self.rnn = nn.LSTM(embed_size, hidden_size, batch_first=True)
        # output
        self.output = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_seq, length):
        # process input
        sorted_len, sorted_idx = torch.sort(length, descending=True)
        input_seq = input_seq[sorted_idx]
        embedded_input = self.embedding(input_seq)
        drop_input = F.dropout(embedded_input, p=self.dropout_rate,
                               training=self.training)

        # RNN forward
        pack_input = pack_padded_sequence(drop_input, sorted_len + 1,
                                          batch_first=True)
        pack_output, _ = self.rnn(pack_input)
        output, _ = pad_packed_sequence(pack_output, batch_first=True)
        _, reversed_idx = torch.sort(sorted_idx)
        output = output[reversed_idx]

        # project output
        drop_output = F.dropout(output, p=self.dropout_rate,
                                training=self.training)
        batch_size, seq_len, hidden_size = drop_output.size()
        logit = self.output(drop_output.view(-1, hidden_size))
        logp = F.log_softmax(logit, dim=-1)
        logp = logp.view(batch_size, seq_len, -1)

        return logp


class LSTMEncoder(nn.Module):

    def __init__(self, vocab_size, embed_size, hidden_size, z_dim, pad_idx):
        super(LSTMEncoder, self).__init__()
        # input
        self.embedding = nn.Embedding(vocab_size, embed_size,
                                      padding_idx=pad_idx)
        # RNN
        self.rnn = nn.LSTM(embed_size, hidden_size, batch_first=True)
        # output
        self.output = nn.Linear(hidden_size * 2, z_dim * 2)

    def forward(self, input_seq, length):
        # embed input
        embedded_input = self.embedding(input_seq)

        # RNN forward
        pack_input = pack_padded_sequence(embedded_input, length,
                                          batch_first=True)
        _, (h, c) = self.rnn(pack_input)

        # produce mu and logvar
        hidden = torch.cat([h, c], dim=-1).squeeze(0)
        mu, logvar = torch.chunk(self.output(hidden), 2, dim=-1)

        return mu, logvar


class RNNVAE(nn.Module):

    def __init__(self, vocab_size, embed_size, time_step, hidden_size, z_dim,
                 dropout_rate, bos_idx, eos_idx, pad_idx):
        super(RNNVAE, self).__init__()
        self.time_step = time_step
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx
        self.pad_idx = pad_idx

        # encoder
        self.encoder = LSTMEncoder(vocab_size, embed_size,
                                   hidden_size, z_dim, pad_idx)
        # decoder
        self.embedding = nn.Embedding(vocab_size, embed_size,
                                      padding_idx=pad_idx)
        self.init_h = nn.Linear(z_dim, hidden_size)
        self.init_c = nn.Linear(z_dim, hidden_size)
        self.rnn = nn.LSTM(embed_size, hidden_size, batch_first=True)
        # output
        self.output = nn.Linear(hidden_size, vocab_size)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(mu)
        return mu + std * eps

    def forward(self, enc_input, dec_input, length):
        # process input
        sorted_len, sorted_idx = torch.sort(length, descending=True)
        enc_input = enc_input[sorted_idx]
        dec_input = dec_input[sorted_idx]

        # encode
        mu, logvar = self.encoder(enc_input, sorted_len)
        z = self.reparameterize(mu, logvar)

        # decode
        embedded_input = self.embedding(dec_input)
        drop_input = F.dropout(embedded_input, p=self.dropout_rate,
                               training=self.training)
        pack_input = pack_padded_sequence(drop_input, sorted_len + 1,
                                          batch_first=True)
        h_0, c_0 = self.init_h(z), self.init_c(z)
        hidden = (h_0.unsqueeze(0), c_0.unsqueeze(0))
        pack_output, _ = self.rnn(pack_input, hidden)
        output, _ = pad_packed_sequence(pack_output, batch_first=True)
        _, reversed_idx = torch.sort(sorted_idx)
        output = output[reversed_idx]

        # project output
        batch_size, seq_len, hidden_size = output.size()
        logit = self.output(output.view(-1, hidden_size))
        logp = F.log_softmax(logit, dim=-1)
        logp = logp.view(batch_size, seq_len, -1)

        return logp, mu, logvar


class pBLSTMLayer(nn.Module):

    def __init__(self, input_dim, hidden_size, factor=2):
        super(pBLSTMLayer, self).__init__()
        self.factor = factor
        self.rnn = nn.LSTM(input_dim * factor, hidden_size,
                           batch_first=True, bidirectional=True)

    def forward(self, input_seq):
        batch_size, time_step, input_dim = input_seq.size()
        input_seq = input_seq.contiguous().view(batch_size,
                                                time_step // self.factor,
                                                input_dim * self.factor)
        output, _ = self.rnn(input_seq)

        return output


class pBLSTMEncoder(nn.Module):

    def __init__(self, vocab_size, embed_size, time_step,
                 hidden_size, z_dim, pad_idx):
        super(pBLSTMEncoder, self).__init__()
        # input
        self.embedding = nn.Embedding(vocab_size, embed_size,
                                      padding_idx=pad_idx)
        # pBLSTM
        self.pBLSTM = nn.Sequential(
            pBLSTMLayer(embed_size + z_dim, hidden_size),
            pBLSTMLayer(hidden_size * 2, hidden_size),
            pBLSTMLayer(hidden_size * 2, hidden_size)
        )
        # ouput
        self.output = nn.Sequential(
            nn.Linear(hidden_size * 2 * time_step // 8, hidden_size * 2),
            nn.Linear(hidden_size * 2, z_dim * 2)
        )

    def forward(self, input_seq, z):
        # process input
        embedded_input = self.embedding(input_seq)
        z_expand = z.unsqueeze(1).expand(embedded_input.size(0),
                                         embedded_input.size(1), z.size(-1))
        new_input = torch.cat([embedded_input, z_expand], dim=-1)

        # pBLSTM forward
        output = self.pBLSTM(new_input)

        # produce mu and logvar
        output = output.contiguous().view(output.size(0), -1)
        mu, logvar = torch.chunk(self.output(output), 2, dim=-1)

        return mu, logvar


class HRNNVAE(nn.Module):

    def __init__(self, vocab_size, embed_size, time_step, hidden_size, z_dim,
                 dropout_rate, bos_idx, eos_idx, pad_idx):
        super(HRNNVAE, self).__init__()
        self.time_step = time_step
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx
        self.pad_idx = pad_idx

        # encoder
        self.encoder_zg = LSTMEncoder(vocab_size, embed_size,
                                      hidden_size, z_dim, pad_idx)
        self.encoder_zl = pBLSTMEncoder(vocab_size, embed_size, time_step,
                                        hidden_size, z_dim, pad_idx)
        # decoder
        self.embedding = nn.Embedding(vocab_size, embed_size,
                                      padding_idx=pad_idx)
        self.rnn = nn.LSTM(embed_size + z_dim * 2,
                           hidden_size, batch_first=True)
        # output
        self.output = nn.Linear(hidden_size, vocab_size)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(mu)
        return mu + std * eps

    def forward(self, enc_input, dec_input, length):
        # process input
        sorted_len, sorted_idx = torch.sort(length, descending=True)
        enc_input = enc_input[sorted_idx]
        dec_input = dec_input[sorted_idx]

        # encoder 1 phrase
        mu_zg, logvar_zg = self.encoder_zg(enc_input, sorted_len)
        zg = self.reparameterize(mu_zg, logvar_zg)

        # encoder 2 phrase
        mu_zl, logvar_zl = self.encoder_zl(enc_input, zg)
        zl = self.reparameterize(mu_zl, logvar_zl)

        # decoder
        embedded_input = self.embedding(dec_input)
        drop_input = F.dropout(embedded_input, p=self.dropout_rate,
                               training=self.training)
        z = torch.cat([zg, zl], dim=-1)
        z_expand = z.unsqueeze(1).expand(embedded_input.size(0),
                                         embedded_input.size(1), z.size(-1))
        new_input = torch.cat([drop_input, z_expand], dim=-1)
        pack_input = pack_padded_sequence(new_input, sorted_len + 1,
                                          batch_first=True)
        packed_output, _ = self.rnn(pack_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        _, reversed_idx = torch.sort(sorted_idx)
        output = output[reversed_idx]

        # output
        batch_size, seq_len, hidden_size = output.size()
        logit = self.output(output.view(-1, hidden_size))
        logp = F.log_softmax(logit, dim=-1)
        logp = logp.view(batch_size, seq_len, -1)

        return logp, mu_zg, logvar_zg, mu_zl, logvar_zl
