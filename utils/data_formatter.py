# -*- coding: utf-8 -*-
from os.path import join

from utils.data_manage import pickle_save, pickle_load
from utils.data_util import get_sent_id_indices_for_long_text, padding
from settings import parameters as para


def translate_dict_on_event_to_on_doc(dict_on_event):
    dict_on_doc = {}
    dict_on_doc_for_event_list = {}
    dict_on_doc_for_token_list = {}
    for event_key in dict_on_event.keys():
        for dict_solo_event in dict_on_event[event_key]:
            doc_id = dict_solo_event['doc_id']
            if doc_id not in dict_on_doc.keys():
                dict_on_doc[doc_id] = []
                dict_on_doc_for_event_list[doc_id] = []
            dict_solo_event['event_type'] = event_key
            # dict_solo_event.pop('doc_id')
            dict_on_doc[doc_id].append(dict_solo_event)
            dict_on_doc_for_event_list[doc_id].append(event_key)
    # sort the dict_solo_event in each doc, and get the event token lists
    for doc_id in dict_on_doc.keys():
        list_sentid = []
        list_dict_solo_event_temp = []
        list_for_event_list_temp = []
        dict_on_doc_for_token_list[doc_id] = []
        for dict_solo_event in dict_on_doc[doc_id]:
            list_sentid.append(dict_solo_event['sent_id'])
        list_sentid = list(set(list_sentid))
        list_sentid.sort()

        for i in range(len(list_sentid)):
            for dict_solo_event in dict_on_doc[doc_id]:
                if list_sentid[i] == dict_solo_event['sent_id']:
                    list_dict_solo_event_temp.append(dict_solo_event)
                    list_for_event_list_temp.append(dict_solo_event['event_type'])
                    dict_on_doc_for_token_list[doc_id].append([dict_solo_event['sent_id'], dict_solo_event['event_mention_tokens']])
                    # break
        dict_on_doc[doc_id] = list_dict_solo_event_temp
        dict_on_doc_for_event_list[doc_id] = list_for_event_list_temp

    return dict_on_doc, dict_on_doc_for_event_list, dict_on_doc_for_token_list


def get_event_rels(dict_on_event, dict_event_relation, dict_relation2id, data_rel2array, start_index, subtype2id, max_length):
    dict_on_doc, dict_on_doc_for_event_list, dict_on_doc_for_token_list = translate_dict_on_event_to_on_doc(dict_on_event)
    for event_subtype in dict_on_event.keys():
        for i in range(len(dict_on_event[event_subtype])):
            dict_solo_event = dict_on_event[event_subtype][i]
        # for dict_solo_event in dict_add_context_on_event[event_subtype]:
            dict_sentid_index_for_all, list_sent_id_indices, target_index, dict_solo_event = get_sent_id_indices_for_long_text(
                dict_on_doc_for_token_list[dict_solo_event['doc_id']], dict_solo_event, max_length)

            doc_id = dict_solo_event['doc_id']
            for rel in dict_event_relation[doc_id].keys():
                rel2id = dict_relation2id[rel]
                j = 0
                # print(self.ori_event_relation[doc_id][rel])
                for event_pair in dict_event_relation[doc_id][rel]:
                    if dict_solo_event['sent_id'] == event_pair[1]:
                        # print(event_pair, dict_sentid_index_for_all.keys())
                        if event_pair[0] in dict_sentid_index_for_all.keys():
                            h_sent_index = dict_sentid_index_for_all[event_pair[0]]
                            data_rel2array[i + start_index][rel2id][j][0] = h_sent_index
                            h_event_subtype = dict_on_doc[doc_id][h_sent_index]['event_type']
                            data_rel2array[i + start_index][rel2id][j][1] = subtype2id[h_event_subtype]
                            j += 1
            # print(subtype, dict_solo_event['sent_id'], data_rel2array[i])

    return data_rel2array


def get_event_context(dict_on_event, max_length):
    dict_add_context_on_event = dict_on_event
    dict_on_doc, dict_on_doc_for_event_list, dict_on_doc_for_token_list = \
        translate_dict_on_event_to_on_doc(dict_add_context_on_event)
    for event_subtype in dict_add_context_on_event.keys():
        for i in range(len(dict_add_context_on_event[event_subtype])):
            dict_solo_event = dict_add_context_on_event[event_subtype][i]
        # for dict_solo_event in dict_add_context_on_event[event_subtype]:
            dict_sentid_index_for_all, list_sent_id_indices, target_index, dict_solo_event = get_sent_id_indices_for_long_text(
                dict_on_doc_for_token_list[dict_solo_event['doc_id']], dict_solo_event, max_length)
            pos2 = len(dict_solo_event['event_mention_tokens']) - 1 + 2

            token_list = []
            pos1_list = []
            pos2_list = []
            trigger_array = []

            for k in range(len(list_sent_id_indices)):
                sent_id_index = list_sent_id_indices[k]
                if sent_id_index == target_index:
                    token_list.extend(['[CLS]'] + dict_solo_event['event_mention_tokens'] + ['[SEP]'])
                    # there is only one sentence index (target_index) in list_sent_id_indices, and len exceeds
                    if len(token_list) > max_length:
                        pos1_list.extend([x for x in range(max_length)])
                        pos2_list.extend([pos2 - 1] * max_length)
                        target_trigger_array = [0] * max_length
                        if type(dict_solo_event['trigger_pos']) == list:
                            for pos_i in range(dict_solo_event['trigger_pos'][0], dict_solo_event['trigger_pos'][1] + 1):
                                if pos_i + 1 < max_length:
                                    target_trigger_array[pos_i + 1] = 1
                        else:
                            if dict_solo_event['trigger_pos'] + 1 < max_length:
                                target_trigger_array[dict_solo_event['trigger_pos'] + 1] = 1
                        trigger_array.extend(target_trigger_array)
                        break
                    len_tokens = len(dict_solo_event['event_mention_tokens']) + 2
                    pos1_list.extend([x for x in range(len_tokens)])
                    pos2_list.extend([pos2 - 1] * len_tokens)
                    target_trigger_array = [0] * len_tokens
                    if type(dict_solo_event['trigger_pos']) == list:
                        for pos_i in range(dict_solo_event['trigger_pos'][0], dict_solo_event['trigger_pos'][1] + 1):
                            target_trigger_array[pos_i + 1] = 1
                    else:
                        target_trigger_array[dict_solo_event['trigger_pos'] + 1] = 1
                    trigger_array.extend(target_trigger_array)
                else:
                    solo_event_tokens = dict_on_doc_for_token_list[dict_solo_event['doc_id']][sent_id_index][1]
                    token_list.extend(solo_event_tokens)
                    len_tokens = len(solo_event_tokens)
                    pos1_list.extend([x for x in range(len_tokens)])
                    pos2_list.extend([len_tokens - 1] * len_tokens)
                    trigger_array.extend([0] * len_tokens)

            # padding
            token_list = padding(token_list, max_length, '[PAD]')
            trigger_array = padding(trigger_array, max_length, 2)
            pos1_list = padding(pos1_list, max_length, max_length - 1)
            pos2_list = padding(pos2_list, max_length, max_length - 1)
            dict_solo_event['context_token_list'] = token_list
            dict_solo_event['pos1_list'] = pos1_list
            dict_solo_event['pos2_list'] = pos2_list
            dict_solo_event['trigger_array'] = trigger_array
            dict_add_context_on_event[event_subtype][i] = dict_solo_event

    return dict_add_context_on_event


def get_event_formal_tokens(dict_on_event, max_length):
    dict_formal_tokens_on_event = dict_on_event
    for event_subtype in dict_formal_tokens_on_event.keys():
        for i in range(len(dict_formal_tokens_on_event[event_subtype])):
            formal_tokens = []
            dict_solo_event = dict_formal_tokens_on_event[event_subtype][i]
            len_token_list = len(dict_solo_event['event_mention_tokens'])
            if len_token_list > max_length:
                if type(dict_solo_event['trigger_pos']) == list:
                    left_end_index = dict_solo_event['trigger_pos'][0] - 1
                    right_begin_index = dict_solo_event['trigger_pos'][1] + 1
                    for pos_i in range(dict_solo_event['trigger_pos'][0], dict_solo_event['trigger_pos'][1] + 1):
                        formal_tokens.append(dict_solo_event['event_mention_tokens'][pos_i])
                else:
                    left_end_index = dict_solo_event['trigger_pos'] - 1
                    right_begin_index = dict_solo_event['trigger_pos'] + 1
                    formal_tokens.append(dict_solo_event['event_mention_tokens'][dict_solo_event['trigger_pos']])
                left_ins = int((max_length - len(formal_tokens)) / 2)
                if left_ins > left_end_index + 1:
                    left_ins = left_end_index + 1
                right_ins = max_length - len(formal_tokens) - left_ins
                if right_begin_index + right_ins > len(dict_solo_event['event_mention_tokens']):
                    right_ins = len(dict_solo_event['event_mention_tokens']) - right_begin_index
                    left_ins = max_length - len(formal_tokens) - right_ins
                for j in range(left_end_index, left_end_index - left_ins, -1):
                    formal_tokens = [dict_solo_event['event_mention_tokens'][j]] + formal_tokens
                for j in range(right_begin_index, right_begin_index + right_ins):
                    formal_tokens = formal_tokens + [dict_solo_event['event_mention_tokens'][j]]
                dict_formal_tokens_on_event[event_subtype][i]['event_mention_tokens'] = formal_tokens
                if type(dict_solo_event['trigger_pos']) == list:
                    dict_solo_event['trigger_pos'][0] -= left_end_index - left_ins + 1
                    dict_solo_event['trigger_pos'][1] -= left_end_index - left_ins + 1
                else:
                    dict_solo_event['trigger_pos'] -= left_end_index - left_ins + 1
                # print(dict_solo_event['trigger'], dict_solo_event['trigger_pos'], dict_solo_event['event_mention_tokens'])

    return dict_formal_tokens_on_event


if __name__ == '__main__':
    raw_maven_datapath = join(para.DATA_INPUT_DIRECTORY, "MAVEN")
    raw_ace_datapath = join(para.DATA_INPUT_DIRECTORY, "ace_2005_td_v7/data/English/")
    raw_kbp_datapath = join(para.DATA_INPUT_DIRECTORY, "TAC_KBP_2017/data/eng/")
    raw_ds_datapath = join(para.DATA_INPUT_DIRECTORY, "Event_Data_DS/wiki_sentence_annotated.with_trigger.tsv")
    raw_wiki_datapath = join(para.DATA_SEMI_DIRECTORY, "DS Extension.tsv")
    dict_ace_datapath = join(para.DATA_INPUT_DIRECTORY, "event_dict_ace_data.pkl")
    dict_kbp_datapath = join(para.DATA_INPUT_DIRECTORY, "event_dict_kbp_data.pkl")
    dict_ds_datapath = join(para.DATA_INPUT_DIRECTORY, "event_dict_ds_data.pkl")
    dict_wiki_datapath = join(para.DATA_INPUT_DIRECTORY, "event_dict_wiki_data.pkl")
    dict_acekbp_datapath = join(para.DATA_INPUT_DIRECTORY, "event_dict_ace-kbp_data.pkl")
    dict_maven_datapath = join(para.DATA_INPUT_DIRECTORY, "event_dict_maven_data.pkl")
    dict_ontoee_datapath = join(para.DATA_INPUT_DIRECTORY, "event_dict_data.pkl")
    dict_ontoee_addcontext_datapath = join(para.DATA_INPUT_DIRECTORY, "event_dict_data_add-context.pkl")
    dict_ontoee_on_doc_datapath = join(para.DATA_INPUT_DIRECTORY, "event_dict_data_on_doc.pkl")
    xlsx_event_class_datapath = join(para.DATA_SEMI_DIRECTORY, "OntoED_Event_Types.xlsx")
    dict_ontoee_train_all_datapath = join(para.DATA_INPUT_DIRECTORY, "Overall/event_dict_train_data.pkl")
    dict_ontoee_valid_all_datapath = join(para.DATA_INPUT_DIRECTORY, "Overall/event_dict_valid_data.pkl")
    dict_ontoee_test_all_datapath = join(para.DATA_INPUT_DIRECTORY, "Overall/event_dict_test_data.pkl")
    dict_ontoee_train_all_form_datapath = join(para.DATA_INPUT_DIRECTORY, "Overall/event_dict_train_form_data.pkl")
    dict_ontoee_valid_all_form_datapath = join(para.DATA_INPUT_DIRECTORY, "Overall/event_dict_valid_form_data.pkl")
    dict_ontoee_test_all_form_datapath = join(para.DATA_INPUT_DIRECTORY, "Overall/event_dict_test_form_data.pkl")
    dict_ontoee_train_low_datapath = join(para.DATA_INPUT_DIRECTORY, "Low-resource/event_dict_train_data.pkl")
    dict_ontoee_valid_low_datapath = join(para.DATA_INPUT_DIRECTORY, "Low-resource/event_dict_valid_data.pkl")
    dict_ontoee_test_low_datapath = join(para.DATA_INPUT_DIRECTORY, "Low-resource/event_dict_test_data.pkl")
    dict_ontoee_on_doc_for_rels_of_eventIns_datapath = join(para.DATA_INPUT_DIRECTORY, "rel_dict_event-instance_pairs_data.pkl")
    dict_ontoee_on_rel_for_eventTypePairs_datapath = join(para.DATA_INPUT_DIRECTORY, "rel_dict_event-class_pairs_data.pkl")
    dict_ontoee_on_rel_for_super_sub_datapath = join(para.DATA_INPUT_DIRECTORY, "rel_dict_super_sub_data.pkl")

    print("loading [dict_super_sub]!")
    dict_super_sub = pickle_load(dict_ontoee_on_rel_for_super_sub_datapath)

    """OntoEvent"""
    print("OntoEvent!")
    dict_ontoee_on_event = pickle_load(dict_ontoee_datapath)
    print("number of event classes: ", len(dict_ontoee_on_event))
    for event_key in dict_ontoee_on_event.keys():
        print(event_key, '\t', len(dict_ontoee_on_event[event_key]))

    dict_ontoee_on_doc, dict_ontoee_on_doc_for_event_list, dict_ontoee_on_doc_for_token_list = translate_dict_on_event_to_on_doc(dict_ontoee_on_event)
    pickle_save(dict_ontoee_on_doc, dict_ontoee_on_doc_datapath)
    print("number of documents: ", len(dict_ontoee_on_doc)) # 4115

    dict_event_pair_relations_on_doc = pickle_load(dict_ontoee_on_doc_for_rels_of_eventIns_datapath)
    dict_potential_pairs_on_rel = pickle_load(dict_ontoee_on_rel_for_eventTypePairs_datapath)







