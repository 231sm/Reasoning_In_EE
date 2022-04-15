# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
# Copyright 2021 Shumin Deng
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Multiple choice fine-tuning: utilities to work with multiple choice tasks of reading comprehension  """


import csv
import glob
import json
import codecs
import logging
import os
from typing import List
import torch

import tqdm

from transformers import PreTrainedTokenizer


logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for multiple choice"""

    def __init__(self, example_id, tokens, triggerL, triggerR, label=None):
        """Constructs a InputExample.
        Args:
            example_id: str. unique id for the example.
            tokens: list of tokens. 
            triggerL: int. beginning position of the trigger
            triggerR: int. endding position of the trigger
            label: (Optional) string. The label of the example. This should be specified for train and valid examples, but not for test examples.
        """
        self.example_id = example_id
        self.tokens = tokens
        self.triggerL = triggerL
        self.triggerR = triggerR
        self.label = label


class InputFeatures(object):
    def __init__(self, example_id, input_ids, input_mask, segment_ids, label, rel_example_ids, rel_label_ids):
        self.example_id = example_id
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label = label
        self.rel_example_ids = rel_example_ids
        self.rel_label_ids = rel_label_ids


class DataProcessor(object):
    """Base class for data converters for multiple choice data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_valid_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the valid set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the test set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()


class OntoEventProcessor(DataProcessor):
    """Processor for the OntoEvent data set."""
    
    def get_train_examples(self, data_dir):
        logger.info("LOOKING AT {} train".format(data_dir))
        return self.create_examples(os.path.join(data_dir,'event_dict_train_data.json'), "train")

    def get_valid_examples(self, data_dir):
        logger.info("LOOKING AT {} valid".format(data_dir))
        return self.create_examples(os.path.join(data_dir,'event_dict_valid_data.json'), "valid")

    def get_test_examples(self, data_dir):
        logger.info("LOOKING AT {} test".format(data_dir))
        return self.create_examples(os.path.join(data_dir,'event_dict_test_data.json'), "test")

    def get_labels(self):
        file_path = LABEL_PATH 
        data = json2dicts(file_path)
        list_label = [key for key in data[0].keys()]
        return list_label
    
    
    
    def create_examples(self, file_path, set_type):
        """Creates examples for the training and valid sets."""
        examples = []
        data = json2dicts(file_path)[0]
        
        for event_type in data.keys():
            for event_instance in data[event_type]:
                dict_tailEvent = {}
                sid = event_instance['sent_id']
                if type(event_instance['sent_id'] != str):
                    sid = str(sid)    
                # e_id = "%s-+-%s-+-%s" % (set_type, event_instance['doc_id'], sid)
                e_id = "%s-+-%s-+-%s" % (event_instance['event_type'], event_instance['doc_id'], sid)
                if (type(event_instance['trigger_pos']) == int):
                    triL = event_instance['trigger_pos']
                    triR = triL
                else:
                    triL = event_instance['trigger_pos'][0]
                    triR = event_instance['trigger_pos'][1]
                examples.append(
                    InputExample(
                        example_id=e_id,
                        tokens=event_instance['event_mention_tokens'],
                        triggerL=triL,
                        triggerR=triR,
                        label=event_instance['event_type'],
                    )
                )
        return examples

    
def json2dicts(jsonFile):
        data = []
        with codecs.open(jsonFile, "r", "utf-8") as f:
            for line in f:
                dic = json.loads(line)
                data.append(dic)
        return data
    
    
def convert_examples_to_features(
    examples: List[InputExample],
    label_list: List[str],
    max_length: int,
    tokenizer: PreTrainedTokenizer,
    pad_token_segment_id=0,
    pad_on_left=False,
    pad_token=0,
    mask_padding_with_zero=True,
) -> List[InputFeatures]:
    """
    Loads a data file into a list of `InputFeatures`
    """
    
    label_map = {label: i for i, label in enumerate(label_list)}
    
    list_example_id = set()
    for (ex_index, example) in tqdm.tqdm(enumerate(examples), desc="convert examples to features"):
        list_example_id.add(example.example_id)
    example_id_map = {example_id: i+1 for i, example_id in enumerate(list_example_id)} # eid counts from 1
    
    relation_map = {'BEFORE': 1, 'AFTER': 2, 'EQUAL': 3, 'CAUSE': 4, 'CAUSEDBY': 5, 'COSUPER': 6, 'SUBSUPER': 7, 'SUPERSUB': 8}
    
    dict_rel2events = json2dicts(RELATION_PATH)[0]
    max_num_eventPair = 0
    for rel in dict_rel2events.keys():
        max_num_eventPair = max(len(dict_rel2events[rel]), max_num_eventPair)
    matrix_rel_example_ids = torch.zeros([len(relation_map), max_num_eventPair, 2], dtype=torch.long)
    matrix_rel_label_ids = torch.zeros([len(relation_map), max_num_eventPair, 2], dtype=torch.int)
    for rel in dict_rel2events.keys():
        if rel not in ['COSUPER', 'SUBSUPER', 'SUPERSUB']:
            rel_id = relation_map[rel] - 1
            list_event_pairs = dict_rel2events[rel]
            for i in range(len(list_event_pairs)):
                if type(list_event_pairs[i][0]['sent_id']) != str:
                    list_event_pairs[i][0]['sent_id'] = str(list_event_pairs[i][0]['sent_id'])
                if type(list_event_pairs[i][1]['sent_id']) != str:
                    list_event_pairs[i][1]['sent_id'] = str(list_event_pairs[i][0]['sent_id'])
                head_event_id = "%s-+-%s-+-%s" % (list_event_pairs[i][0]['event_type'], list_event_pairs[i][0]['doc_id'], list_event_pairs[i][0]['sent_id'])
                tail_event_id = "%s-+-%s-+-%s" % (list_event_pairs[i][1]['event_type'], list_event_pairs[i][1]['doc_id'], list_event_pairs[i][1]['sent_id'])
                if head_event_id in example_id_map.keys() and tail_event_id in example_id_map.keys():
                    matrix_rel_example_ids[rel_id][i][0] = example_id_map[head_event_id]
                    matrix_rel_example_ids[rel_id][i][1] = example_id_map[tail_event_id]
                    matrix_rel_label_ids[rel_id][i][0] = label_map[list_event_pairs[i][0]['event_type']]
                    matrix_rel_label_ids[rel_id][i][1] = label_map[list_event_pairs[i][1]['event_type']]
        else:
            for rel in ['COSUPER']: # , 'SUBSUPER', 'SUPERSUB'
                rel_id = relation_map[rel] - 1
                list_event_pairs = dict_rel2events[rel]
                for i in range(len(list_event_pairs)):
                    matrix_rel_label_ids[rel_id][i][0] = label_map[list_event_pairs[i][0]]
                    matrix_rel_label_ids[rel_id][i][1] = label_map[list_event_pairs[i][1]]
                
    features = []
    for (ex_index, example) in tqdm.tqdm(enumerate(examples), desc="convert examples to features"):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))
        
        # the token inputs with trigger mark
        textL = tokenizer.tokenize(" ".join(example.tokens[:example.triggerL]))
        textTrg = tokenizer.tokenize(" ".join(example.tokens[example.triggerL:example.triggerR]))
        textR = tokenizer.tokenize(" ".join(example.tokens[example.triggerR:]))
        text = textL + ['[<trigger>]'] + textTrg + ['[</trigger>]'] + textR
        # # the raw token inputs
        # text = tokenizer.tokenize(" ".join(example.tokens[:]))
        inputs = tokenizer.encode_plus(
            text, add_special_tokens=True, max_length=max_length, return_token_type_ids=True
        )
        
        if "num_truncated_tokens" in inputs and inputs["num_truncated_tokens"] > 0:
            logger.info(
                "Attention! You are cropping tokens."
            )

        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]

        # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
        # Zero-pad up to the sequence length.
        padding_length = max_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
            token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_length
        assert len(attention_mask) == max_length
        assert len(token_type_ids) == max_length
        
        example_id = example_id_map[example.example_id]
        label = label_map[example.label]
        
        rel_example_ids = torch.zeros([len(relation_map), max_num_eventPair], dtype=torch.long)
        rel_label_ids = torch.ones([len(relation_map), len(label_map)], dtype=torch.int) * len(label_map)
        for i in range(len(relation_map)):
            k = 0
            for j in range(len(label_map)):
                if matrix_rel_example_ids[i][j][1] == example_id:
                    rel_example_ids[i][j] = matrix_rel_example_ids[i][j][0]
                    if matrix_rel_label_ids[i][j][0] not in rel_label_ids[i]:
                        rel_label_ids[i][k] = matrix_rel_label_ids[i][j][0]
                        k += 1
                    
        if ex_index < 2:
            logger.info("*** Example ***")
            logger.info("example_id: {}".format(example.example_id))
            logger.info("input_ids: {}".format(" ".join(map(str, input_ids))))
            logger.info("attention_mask: {}".format(" ".join(map(str, attention_mask))))
            logger.info("token_type_ids: {}".format(" ".join(map(str, token_type_ids))))
            logger.info("label: {}".format(label))
            logger.info("rel_example_ids: {}".format(" ".join(map(str, rel_example_ids))))
            logger.info("rel_label_ids: {}".format(" ".join(map(str, rel_label_ids))))

        features.append(InputFeatures(example_id=example_id, input_ids=input_ids, input_mask=attention_mask, segment_ids=token_type_ids, label=label, rel_example_ids=rel_example_ids, rel_label_ids=rel_label_ids))

    return features



processors = {"ontoevent": OntoEventProcessor} # other dataset can also be used here

MULTIPLE_CHOICE_TASKS_NUM_LABELS = {"ontoevent", 100} # other dataset can also be used here

LABEL_PATH = "../OntoEvent/event_dict_test_data.json"
# # file path for the json data contains all labels, such as './event_dict_train_data.json'
RELATION_PATH = "../OntoEvent/event_relation.json"
# # file path for the json data contains all relations, such as './event_relation.json'