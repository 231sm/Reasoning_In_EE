# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

from pytorch_pretrained_bert import BertModel
from settings import parameters as para


class Embedding(nn.Module):

    def __init__(self, word_vec_mat, concate_dim, max_length=para.MAX_LENGTH,
                 word_embedding_dim=para.SIZE_WORDVEC, pos_embedding_dim=para.SIZE_POSVEC):
        nn.Module.__init__(self)

        self.concat_dim = concate_dim
        self.max_length = max_length
        self.word_embedding_dim = word_embedding_dim
        self.pos_embedding_dim = pos_embedding_dim
        """Word embedding"""
        pad = torch.zeros(1, word_embedding_dim)
        word_vec_mat = torch.from_numpy(word_vec_mat)
        self.word_embedding = nn.Embedding(word_vec_mat.size()[0] + 1, self.word_embedding_dim)
        self.word_embedding.weight.data.copy_(torch.cat((word_vec_mat, pad), 0))
        """Position Embedding"""
        self.pos1_embedding = nn.Embedding(max_length, pos_embedding_dim, padding_idx=0)
        self.pos2_embedding = nn.Embedding(max_length, pos_embedding_dim, padding_idx=0)

    def forward(self, inputs):
        word = inputs['word']
        pos1 = inputs['pos1']
        pos2 = inputs['pos2']
        x = torch.cat((self.word_embedding(word),
                       self.pos1_embedding(pos1),
                       self.pos2_embedding(pos2)),
                      self.concat_dim) # self.concat_dim = 2, 3

        if para.ENCODER_MODEL == "BERT":
            if para.CUDA:
                bert = BertModel.from_pretrained('bert-base-cased').cuda()
            else:
                bert = BertModel.from_pretrained('bert-base-cased')
            encoded_layers, _ = bert(inputs['word'], output_all_encoded_layers=False)
            x = encoded_layers[0]
        return x
        # x.size() -> (#batch, #instances_for_all_classes, #max_word_in_a_sent, #embedding)

