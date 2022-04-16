from __future__ import print_function
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers import BertPreTrainedModel,BertModel

logger = logging.getLogger(__name__)

relation_map = {'BEFORE': 1, 'AFTER': 2, 'EQUAL': 3, 'CAUSE': 4, 'CAUSEDBY': 5, 'COSUPER': 6, 'SUBSUPER': 7, 'SUPERSUB': 8}

class OntoED(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.maxpooling = nn.MaxPool1d(128)
        self.re_classifier = nn.Linear(config.hidden_size*4, len(relation_map)+1)
        # some hyperparameters
        self.ratio_proto_emb = 0.5 
        self.ratio_loss_ed = 1 
        self.ratio_loss_op = 4 
        self.ratio_loss_ol = 1
        self.proto = Proto(config.num_labels, config.hidden_size, self.ratio_proto_emb)
        self.relation = Relation(config.hidden_size)
    
    def get_event_re_task(self, instance_embedding, example_ids, rel_example_ids):
        batch_size = instance_embedding.size(0)
        hidden_size = instance_embedding.size(1)
        num_rel = rel_example_ids.size(1)
        inputs_ere = torch.zeros([batch_size*(batch_size-1), hidden_size*4], dtype=torch.float).to(torch.device("cuda"))
        labels_ere = torch.zeros([batch_size*(batch_size-1)], dtype=torch.long).to(torch.device("cuda"))
        
        if torch.sum(rel_example_ids) == 0:
            return inputs_ere, None
        
        count_example_pair = 0
        for i in range(batch_size):
            for j in range(batch_size):
                if i != j:
                    inputs_ere[count_example_pair] = self.get_embedding_interaction(instance_embedding[j], instance_embedding[i])
                    for k in range(num_rel):
                        if example_ids[j] in rel_example_ids[i][k]:
                            labels_ere[count_example_pair] = k + 1
                            break
                    count_example_pair += 1
        
        return inputs_ere, labels_ere
      
    def get_embedding_interaction(self, t1, t2):
        return torch.cat([t1, t2, torch.mul(t1, t2), t1 - t2], dim=0)
         
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None, labels=None, example_ids=None, rel_example_ids=None, rel_label_ids=None): 
        batch_size = input_ids.size(0)
        
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        
        sequence_output = outputs[0]
        pooled_output = self.maxpooling(sequence_output.transpose(1, 2)).contiguous().view(batch_size, self.config.hidden_size)
        pooled_output = F.relu(pooled_output)
        instance_embedding = self.dropout(pooled_output) # [batch_size, hidden_size]
        
        inputs_ere, labels_ere = self.get_event_re_task(instance_embedding, example_ids, rel_example_ids)
        logits_ere = self.re_classifier(inputs_ere) 
        # _, pred_ere = torch.max(logits_ere, dim=1)
        
        torch.autograd.set_detect_anomaly(True)
        
        loss_er, relation_embedding = self.relation()
        
        logits, loss_ol, proto_embedding = self.proto(instance_embedding, relation_embedding, rel_label_ids, labels)
        outputs = (logits,) + outputs[2:]
        
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss_ed = loss_fct(logits, labels)
            if labels_ere is not None:
                loss_ere = loss_fct(logits_ere, labels_ere)
                loss_op = self.ratio_loss_ed*loss_ed + (1-self.ratio_loss_ed)*loss_ere
            else:
                loss_op = loss_ed
            loss = self.ratio_loss_op*loss_op + self.ratio_loss_ol*loss_ol + loss_er
            outputs = (loss,) + outputs
            
        return outputs


class Proto(nn.Module):
    def __init__(self, proto_size, hidden_size, proto_emb_ratio):
        super(Proto, self).__init__()
        self.prototypes = nn.Embedding(proto_size, hidden_size).to(torch.device("cuda"))
        self.classifier = nn.Linear(hidden_size, proto_size)
        self.proto_size = proto_size
        self.hidden_size = hidden_size
        self.proto_emb_ratio = proto_emb_ratio
    
    def __dist__(self, x, y, dim):
        dist = torch.pow(x - y, 2).sum(dim)
        # dist = torch.where(torch.isnan(dist), torch.full_like(dist, 1e-8), dist)
        return dist
    
    def __batch_dist__(self, S, Q):
        return self.__dist__(S.unsqueeze(0), Q.unsqueeze(1), 2)
    
    def get_proto_embedding(self):
        proto_embedding = self.prototypes(torch.tensor(range(0, self.proto_size)).to(torch.device("cuda")))
        return proto_embedding # [num_labels, hidden_size]
    
    def get_similarity(self, h_index, r_index, t_index, relation_embedding, proto_embedding): 
        if h_index != self.proto_size:
            p_h = proto_embedding[h_index].unsqueeze(0)
        else:
            p_h = torch.zeros([1, self.hidden_size]).to(torch.device("cuda"))
        m_r = relation_embedding[r_index]
        p_t = proto_embedding[t_index].unsqueeze(0)
        return torch.sigmoid(torch.matmul(torch.matmul(p_h, m_r), p_t.transpose(0, 1)))
    
    def onto_learn(self, proto_embedding, relation_embedding, rel_label_ids, labels):
        proto_embedding = proto_embedding
        batch_size = rel_label_ids.size(0)
        num_rel = rel_label_ids.size(1)
        labels_unique, index_list = torch.unique(labels, return_inverse=True)
        
        num_unique = labels_unique.size(0)
        
        logits_ol = torch.zeros([num_unique, num_rel, self.proto_size, 2], dtype=torch.float).to(torch.device("cuda"))
        logits_ol[:][:][:][0] = 1.0
        labels_ol = torch.zeros([num_unique, num_rel, self.proto_size], dtype=torch.long).to(torch.device("cuda"))

        if torch.sum(rel_label_ids) == 0:
            return logits_ol, None, proto_embedding
        for k in range(num_unique):
            index = index_list[k]
            if torch.sum(rel_label_ids[index]) == 0:
                continue
            label_id = labels_unique[k].item()
            num_prop = 0
            proto_prop_embedding = torch.zeros([1, self.hidden_size], dtype=float).to(torch.device("cuda"))
            for i in range(num_rel):
                if torch.sum(rel_label_ids[index][i]) == 0:
                    continue
                for j in range(self.proto_size):
                    head_label_id = rel_label_ids[index][i][j]
                    if head_label_id != self.proto_size:
                        proto_prop_embedding = proto_prop_embedding + torch.matmul(self.get_proto_embedding()[head_label_id].unsqueeze(0), relation_embedding[i])
                        num_prop += 1
                        labels_ol[k][i][j] = 1
                        logits_ol[k][i][j][1] = self.get_similarity(head_label_id, i, label_id, relation_embedding, self.get_proto_embedding())
                        logits_ol[k][i][j][0] = 1 - logits_ol[k][i][j][1]
                    
            if num_prop != 0:
                proto_prop_embedding = proto_prop_embedding / num_prop
                proto_embedding[label_id] = (self.proto_emb_ratio*proto_embedding[label_id].unsqueeze(0) + (1 - self.proto_emb_ratio)*proto_prop_embedding).squeeze(0)                
        logits_ol = logits_ol.view(-1, 2)
        labels_ol = labels_ol.view(-1)

        return logits_ol, labels_ol, proto_embedding
          
        
    def forward(self, instance_embedding, relation_embedding, rel_label_ids, labels):
        proto_embedding = self.get_proto_embedding()
        logits_ol, labels_ol, proto_embedding = self.onto_learn(proto_embedding, relation_embedding, rel_label_ids, labels)
        
        if labels_ol is not None:
            loss_fct = CrossEntropyLoss()
            loss_ol = loss_fct(logits_ol, labels_ol)
        else:
            loss_ol = 0
        
        logits = -self.__batch_dist__(proto_embedding, instance_embedding)
        
        return logits, loss_ol, proto_embedding


class Relation(nn.Module):
    def __init__(self, hidden_size):
        super(Relation, self).__init__()
        self.relation_before = nn.Embedding(hidden_size, hidden_size).to(torch.device("cuda"))
        self.relation_after = nn.Embedding(hidden_size, hidden_size).to(torch.device("cuda"))
        self.relation_equal = nn.Embedding(hidden_size, hidden_size).to(torch.device("cuda"))
        self.relation_cause = nn.Embedding(hidden_size, hidden_size).to(torch.device("cuda"))
        self.relation_causedby = nn.Embedding(hidden_size, hidden_size).to(torch.device("cuda"))
        self.relation_cosuper = nn.Embedding(hidden_size, hidden_size).to(torch.device("cuda"))
        self.relation_subsuper = nn.Embedding(hidden_size, hidden_size).to(torch.device("cuda"))
        self.relation_supersub = nn.Embedding(hidden_size, hidden_size).to(torch.device("cuda"))
        self.identity = torch.cat((torch.ones(int(hidden_size -hidden_size/4)), torch.zeros(int(hidden_size/4))),0).unsqueeze(0).to(torch.device("cuda"))
        self.hidden_size = hidden_size
        self.ratio_sub_op = 0.5
        self.ratio_inver_op = 0.5
        self.ratio_trans_op = 0.5
    
    def split_relation_embedding(self, embedding):
        """split relation embedding
           embedding: [hidden_size, hidden_size]
        """
        assert self.hidden_size % 4 == 0
        num_scalar = self.hidden_size // 2
        num_block = self.hidden_size // 4
        
        if len(embedding.size()) == 2:
            embedding_scalar = embedding[:, 0:num_scalar]
            embedding_x = embedding[:, num_scalar:-num_block]
            embedding_y = embedding[:, -num_block:]
        elif len(embedding.size()) == 3:
            embedding_scalar = embedding[:, :, 0:num_scalar]
            embedding_x = embedding[:, :, num_scalar:-num_block]
            embedding_y = embedding[:, :, -num_block:]
        else:
            raise NotImplementedError

        return embedding_scalar, embedding_x, embedding_y
    
    
    def embedding_similarity(self, head=None, tail=None):
        """calculate the similrity between two matrices in relation constraint
           head: [hidden_size, hidden_size]
           tail: [hidden_size, hidden_size]
        """
        A_scalar, A_x, A_y = self.split_relation_embedding(head)
        B_scalar, B_x, B_y = self.split_relation_embedding(tail)

        similarity = torch.cat([(A_scalar - B_scalar)**2, (A_x - B_x)**2, (A_x - B_x)**2, (A_y - B_y)**2, (A_y - B_y)**2 ], dim=1)
        fro_norm = torch.sqrt(torch.sum(similarity, dim=1))
        
        # rescale the probability
        probability = torch.mean( (torch.max(fro_norm) - fro_norm) / (torch.max(fro_norm) - torch.min(fro_norm)) )
        return probability
    
    def forward(self):
        relation_before = self.relation_before(torch.tensor(range(0, self.hidden_size)).to(torch.device("cuda")))
        relation_after = self.relation_after(torch.tensor(range(0, self.hidden_size)).to(torch.device("cuda")))
        relation_equal = self.relation_equal(torch.tensor(range(0, self.hidden_size)).to(torch.device("cuda")))
        relation_cause = self.relation_cause(torch.tensor(range(0, self.hidden_size)).to(torch.device("cuda")))
        relation_causedby = self.relation_causedby(torch.tensor(range(0, self.hidden_size)).to(torch.device("cuda")))
        relation_cosuper = self.relation_cosuper(torch.tensor(range(0, self.hidden_size)).to(torch.device("cuda")))
        relation_subsuper = self.relation_subsuper(torch.tensor(range(0, self.hidden_size)).to(torch.device("cuda")))
        relation_supersub = self.relation_supersub(torch.tensor(range(0, self.hidden_size)).to(torch.device("cuda")))
        
        bias = 5e-5
        loss_sub = -torch.log(self.embedding_similarity(relation_cause, relation_before) + bias)
        loss_inver = -torch.log(self.embedding_similarity(relation_subsuper*relation_supersub, self.identity) + bias)
        loss_inver = loss_inver -torch.log(self.embedding_similarity(relation_before*relation_after, self.identity) + bias)
        loss_inver = loss_inver -torch.log(self.embedding_similarity(relation_cause*relation_causedby, self.identity)+ bias)
        loss_trans = -torch.log(self.embedding_similarity(relation_subsuper*relation_subsuper, relation_subsuper) + bias)
        loss_trans = loss_trans -torch.log(self.embedding_similarity(relation_supersub*relation_supersub, relation_supersub) + bias)
        loss_trans = loss_trans -torch.log(self.embedding_similarity(relation_cosuper*relation_cosuper, relation_cosuper) + bias)
        loss_trans = loss_trans -torch.log(self.embedding_similarity(relation_before*relation_before, relation_before) + bias)
        loss_trans = loss_trans -torch.log(self.embedding_similarity(relation_after*relation_after, relation_after) + bias)
        loss_trans = loss_trans -torch.log(self.embedding_similarity(relation_equal*relation_equal, relation_equal) + bias)
        
        loss = self.ratio_sub_op*loss_sub + self.ratio_inver_op*loss_inver + self.ratio_trans_op*loss_trans
        loss = torch.sigmoid(loss)
        
        relation_embedding = torch.stack((
            relation_before, 
            relation_after, 
            relation_equal, 
            relation_cause, 
            relation_causedby, 
            relation_cosuper, 
            relation_subsuper, 
            relation_supersub
        ), dim=0) # [relation_size, hidden_size, hidden_size]
        
        return loss, relation_embedding
