# -*- coding: utf-8 -*-
import torch.nn as nn
from pytorch_pretrained_bert import BertModel

from models.network_kit import encoder as enc
from models.network_kit import embedding as emb
from settings import parameters as para


class BaselineSentenceEncoder(nn.Module):

    def __init__(self, word_vec_mat, concat_dim, encoder_name):
        nn.Module.__init__(self)
        self.encoder_name = encoder_name
        self.concat_dim = concat_dim    # 2
        self.embedding = emb.Embedding(word_vec_mat, concat_dim)
        self.encoder = enc.Encoder()

        self.fc_trigger = nn.Sequential(nn.Dropout(para.DROPOUT_RATE),
            nn.Linear(para.SIZE_EMB_WORD, para.SIZE_TRIGGER_LABEL, bias=True), nn.ReLU(), )

    def forward(self, inputs):
        x = self.embedding(inputs)
        # x.size() -> (#batch * #instances_for_all_classes, #max_word_in_a_sent, #embedding)
        logits_inputs_trigger = self.fc_trigger(x)
        # logits_inputs_trigger.size() -> (#batch * #instances_for_all_classes, #max_word_in_a_sent, SIZE_TRIGGER_LABEL)
        pred_inputs_trigger = logits_inputs_trigger.argmax(-1)
        # pred_inputs_trigger.size() -> (#batch * #instances_for_all_classes, #max_word_in_a_sent)
        x = self.encoder(x)
        #  x.size() -> (#batch * #instances_for_all_classes, #hidden)
        if self.encoder_name == "BERT":
            bert = BertModel.from_pretrained('bert-base-cased')
            encoded_layers, _ = bert(inputs['word'])
            x = encoded_layers[-1]
        return logits_inputs_trigger, pred_inputs_trigger, x







