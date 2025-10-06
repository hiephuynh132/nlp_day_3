import torch
import torch.nn as nn


import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_layers,
                 bidirectional, dropout, pad_idx):
        """
        @param vocab_size (int)
        @param embedding_dim (int)
        @param hidden_dim (int)
        @param n_layers (int)
        @param bidirectional (bool)
        @param dropout (float)
        @param pad_idx (int)
        """
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=pad_idx)

        self.rnn = nn.LSTM(embedding_dim,
                           hidden_dim,
                           num_layers=n_layers,
                           bidirectional=bidirectional,
                           dropout=dropout)

        self.fc = nn.Linear(hidden_dim * 2, 1)

        self.dropout = nn.Dropout(dropout)

    def forward(self, text, text_lengths):
        """
        @param text (torch.Tensor): shape = [sent len, batch size]
        @param text_lengths (torch.Tensor): shape = [batch size]
        @return
        """
        # text = [sent len, batch size]
        embedded = self.dropout(self.embedding(text))

        # embedded = [sent len, batch size, emb dim]

        # pack sequence
        # lengths need to be on CPU!
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded, text_lengths.to('cpu'))

        packed_output, (hidden, cell) = self.rnn(packed_embedded)

        # unpack sequence
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(
            packed_output)

        # output = [sent len, batch size, hid dim * num directions]
        # output over padding tokens are zero tensors

        # hidden = [num layers * num directions, batch size, hid dim]
        # cell = [num layers * num directions, batch size, hid dim]

        # concat the final forward (hidden[-2,:,:]) and backward (hidden[-1,:,:]) hidden layers
        # and apply dropout

        hidden = self.dropout(
            torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))

        # hidden = [batch size, hid dim * num directions]

        return self.fc(hidden)


class SimpleRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, pad_idx):
        """
        @param vocab_size (int): kích thước vocab
        @param embedding_dim (int): số chiều embedding
        @param hidden_dim (int): số chiều hidden state
        @param pad_idx (int): index của token <pad>
        """
        super().__init__()

        # embedding layer
        """
        num_embeddings (int) – size of the dictionary of embeddings
        embedding_dim (int) – the size of each embedding vector
        """
        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=pad_idx)

        # RNN 1 layer
        self.rnn = nn.RNN(embedding_dim, hidden_dim)

        # fully connected layer
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, text, text_lengths):
        """
        @param text (torch.Tensor): [sent len, batch size]
        @param text_lengths (torch.Tensor): [batch size] độ dài thực tế của từng câu
        @return output (torch.Tensor): [batch size, 1]
        """
        # text = [sent len, batch size]
        embedded = self.embedding(text)
        # embedded = [sent len, batch size, emb dim]

        # pack sequence -> RNN bỏ qua <pad>
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded, text_lengths.cpu(), enforce_sorted=False)

        packed_output, hidden = self.rnn(packed_embedded)
        # packed_output chứa output đã pack, thường không dùng nếu chỉ lấy hidden cuối
        # hidden = [1, batch size, hid dim]  (1 layer, not bidirectional)

        # lấy hidden state cuối cùng
        hidden = hidden.squeeze(0)  # [batch size, hid dim]

        return self.fc(hidden)      # [batch size, 1]
