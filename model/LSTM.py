#!/usr/bin/env python
# encoding: utf-8
from torch import nn
import torch
import torch.nn.functional as F


class LSTM(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim, batch_size=1):
        super(LSTM, self).__init__()
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(input_dim, embedding_dim)

        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        self.fc = nn.Linear(hidden_dim, output_dim)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.zeros(1, self.batch_size, self.hidden_dim).cuda(),
                torch.zeros(1, self.batch_size, self.hidden_dim).cuda())

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)

        # (seq_len, batch, input_size)
        output, self.hidden = self.lstm(embeds, self.hidden)

        assert torch.equal(output[-1, :, :], self.hidden[0].squeeze(0))

        return self.fc(self.hidden[0].squeeze(0))
