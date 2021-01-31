# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

from settings import parameters as para


class Encoder(nn.Module):
    def __init__(self, max_length=para.MAX_LENGTH, word_embedding_dim=para.SIZE_WORDVEC,
                 pos_embedding_dim=para.SIZE_POSVEC, hidden_size=para.SIZE_HIDDEN, encoder_name=para.ENCODER_MODEL):
        nn.Module.__init__(self)

        self.encoder_name = encoder_name
        self.max_length = max_length
        self.hidden_size = hidden_size
        self.embedding_dim = word_embedding_dim + 2 * pos_embedding_dim
        self.conv = nn.Conv1d(self.embedding_dim, self.hidden_size, kernel_size=2, stride=1, padding=2, bias=True)
        self.pool = nn.MaxPool1d(max_length)
        self.bilstm_model = nn.LSTM(self.embedding_dim, hidden_size // 2, num_layers=1, bidirectional=True)

    def forward(self, inputs):
        if self.encoder_name == "CNN":
            return self.cnn(inputs)
        elif self.encoder_name == "LSTM":
            return self.bilstm(inputs)

    def cnn(self, inputs):
        num_all_instances = inputs.size(1)
        inputs = inputs.view(-1, inputs.size(2), inputs.size(3))
        x = self.conv(inputs.transpose(1, 2))
        x = self.pool(x)
        x = F.relu(x)
        return x.squeeze(2).view(-1, num_all_instances, self.hidden_size)
        # (#batch, #instances_for_all_classes, #hidden)

    def bilstm(self, inputs):
        num_all_instances = inputs.size(1)
        inputs = inputs.view(-1, inputs.size(2), inputs.size(3))
        inputs = inputs.transpose(0, 1)
        output, (hn, cn) = self.bilstm_model(inputs)
        return torch.tanh(hn.transpose(0, 1).contiguous().view(-1, num_all_instances, self.hidden_size))


