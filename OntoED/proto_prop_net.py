# -*- coding: utf-8 -*-
import torch
from torch import nn
from torch.autograd import Variable
import numpy as np

from models.model_base import LowResEDModel
from settings import parameters as para


class ProtoPropNet(LowResEDModel):

    def __init__(self, support_sentence_encoder, query_sentence_encoder, hidden_size=para.SIZE_HIDDEN):
        LowResEDModel.__init__(self, support_sentence_encoder, query_sentence_encoder)
        self.hidden_size = hidden_size
        self.fc = nn.Linear(hidden_size, hidden_size)

    def __dist__(self, x, y, dim):
        return (torch.pow(x - y, 2)).sum(dim)

    def __batch_dist__(self, S, Q):
        return self.__dist__(S.unsqueeze(1), Q.unsqueeze(2), 3)

    def forward(self, support, query, scope_support, scope_query, N, R, K, Q):
        '''
        support: Inputs of the support set. # (B, N_K, D)
        query: Inputs of the query set. # (B, N_Q, D)
        scope_support. # (B, N, 2)
        scope_query. # (B, N, 2)
        N: Num of classes
        R: Ratio of instances for each class
        K: Max Num of instances for each class in the support set
        Q: Max Num of instances for each class in the query set
        '''
        logits_support_trigger, pred_support_trigger, support = self.support_sentence_encoder(support)
        logits_query_trigger, pred_query_trigger, query = self.query_sentence_encoder(query)
        D = support.size(-1)  # D: hidden size
        B = support.size(0)

        support_rebuilt = Variable(torch.from_numpy(np.zeros((B, N, D), dtype=np.float32).astype(np.float32)).float())

        for i in range(B):
            for j in range(N):
                begin_index = scope_support[i][j][0]
                end_index = scope_support[i][j][1]
                support_rebuilt[i][j] = torch.mean(support[i][begin_index:end_index+1])

        if para.CUDA:
            support_rebuilt = support_rebuilt.cuda()

        logits = -self.__batch_dist__(support_rebuilt, query)
        _, pred = torch.max(logits.view(-1, N), 1)

        return logits, pred, logits_support_trigger, pred_support_trigger, logits_query_trigger, pred_query_trigger




