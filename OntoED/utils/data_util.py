# -*- coding: utf-8 -*-
from os.path import join

from settings import parameters as para
from utils.data_manage import pickle_load, pickle_save


def padding(list_items, max_length, pad_item):
    if len(list_items) < max_length:
        for i in range(len(list_items), max_length):
            list_items.append(pad_item)
            # if type(list_items) == list:
            #     list_items.append(pad_item)
            # else:
            #     list_items[i] = pad_item
    return list_items


def get_sent_id_indices_for_long_text(multi_token_list, dict_solo_event, max_length):
    list_sent_id = []
    dict_sentid_index_for_all = {}
    list_solo_event_tokens = []
    i = 0
    for [sent_id, solo_event_tokens] in multi_token_list:
        # there is some mistake of sent_id annotation in dict_solo_event
        if solo_event_tokens == dict_solo_event['event_mention_tokens'] and sent_id != dict_solo_event['sent_id']:
            dict_solo_event['sent_id'] = sent_id
        # there is some mistake of event_mention_tokens annotation in dict_solo_event
        if set(solo_event_tokens) > set(dict_solo_event['event_mention_tokens'][:-3]):
            dict_solo_event['event_mention_tokens'] = solo_event_tokens
        elif set(dict_solo_event['event_mention_tokens']) > set(solo_event_tokens[:-3]):
            solo_event_tokens = dict_solo_event['event_mention_tokens']
        list_sent_id.append(sent_id)
        dict_sentid_index_for_all[sent_id] = i
        i += 1
        list_solo_event_tokens.append(solo_event_tokens)

    max_len = len(list_sent_id)
    target_sent_id = dict_solo_event['sent_id']
    target_index = list_sent_id.index(target_sent_id)
    # # Considering there are same sent_ids in a document
    # if list_solo_event_tokens[target_index] != dict_solo_event['event_mention_tokens'] and len(set(list_sent_id)) < len(list_sent_id):
    #     while list_solo_event_tokens[target_index] != dict_solo_event['event_mention_tokens']:
    #         target_index = list_sent_id.index(target_sent_id, target_index + 1)

    len_current = len(dict_solo_event['event_mention_tokens']) + 2
    list_left_sent_id_index = []
    list_right_sent_id_index = []
    i = 1
    while len_current < max_length and (target_index - i > 0 or target_index + i < max_len - 1) and (len(list_left_sent_id_index) + len(list_right_sent_id_index) + 1 < max_len):
        left_index = target_index - i
        right_index = target_index + i
        if left_index >= 0:
            len_current += len(list_solo_event_tokens[left_index])
            if len_current > max_length:
                break
            else:
                list_left_sent_id_index.append(left_index)
        if right_index < max_len:
            len_current += len(list_solo_event_tokens[right_index])
            if len_current > max_length:
                break
            else:
                list_right_sent_id_index.append(right_index)
        i += 1

    list_left_sent_id_index.reverse()
    list_sent_id_indices = list_left_sent_id_index + [target_index] + list_right_sent_id_index

    return dict_sentid_index_for_all, list_sent_id_indices, target_index, dict_solo_event


def get_token_lists_for_docs(dict_on_doc):
    dict_token_list_on_doc = {}
    for doc_id in dict_on_doc.keys():
        dict_token_list_on_doc[doc_id] = []
        for dict_solo_event in dict_on_doc[doc_id]:
            token_list = ['[CLS]'] + dict_solo_event['event_mention_tokens'] + ['[SEP]']
            # if len(token_list) < para.MAX_LENGTH:
            #     token_list = padding(token_list, para.MAX_LENGTH, '[PAD]')
            dict_token_list_on_doc[doc_id].extend(token_list)
    return dict_token_list_on_doc


if __name__ == "__main__":
    dict_ontoee_datapath = join(para.DATA_INPUT_DIRECTORY, "event_dict_data.pkl")
    dict_ontoee_addcontext_datapath = join(para.DATA_INPUT_DIRECTORY, "event_dict_data_add-context.pkl")
    dict_ontoee_on_doc_datapath = join(para.DATA_INPUT_DIRECTORY, "event_dict_data_on_doc.pkl")
    dict_wordVec_datapath = join(para.DATA_INPUT_DIRECTORY, "dict_wordVec.pkl")

    dict_ontoee_on_event = pickle_load(dict_ontoee_datapath)
    dict_ontoee_on_doc = pickle_load(dict_ontoee_on_doc_datapath)
    dict_token_list_on_doc = get_token_lists_for_docs(dict_ontoee_on_doc)

    """wordVec"""
    dict_wordVec = pickle_load(dict_wordVec_datapath)


