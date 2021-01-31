# -*- coding: utf-8 -*-
import sys
from os.path import join
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7,8,9"

from settings import parameters as para
from utils.data_loader import DataLoder
from utils.data_encoder import BaselineSentenceEncoder
from models.model_base import LowResEDFramework, OverallEDFramework
from models.proto_prop_net import ProtoPropNet

encoder_model = para.ENCODER_MODEL
metric_model = para.METRIC_MODEL
model_name = para.PROJECT_NAME + "_" + encoder_model + "-" + metric_model
B = para.SIZE_BATCH  # Batch size
N = para.NUM_CLASS
N_for_train = N
N_for_eval = N
R = para.RATIO_FOR_INSTANCES
R_for_train = R
R_for_eval = R
K = para.MAX_NUM_CLASS_INSTANCE  # Max Num of instances for each class in the support set
Q = para.MAX_NUM_CLASS_INSTANCE  # Max Num of instances for each class in the query set
# K = para.RATIO_FOR_SUPPORT  # Ratio of instances for each class in the support set
# Q = para.RATIO_FOR_QUERY  # Ratio of instances for each class in the query set

noise_rate = 0
if len(sys.argv) > 1:
    model_name = sys.argv[1]
if len(sys.argv) > 2:
    N = int(sys.argv[2])
if len(sys.argv) > 3:
    K = int(sys.argv[3])
if len(sys.argv) > 4:
    noise_rate = float(sys.argv[4])

print("Model: {}".format(model_name))

dict_ontoee_datapath = join(para.DATA_INPUT_DIRECTORY, "event_dict_data.pkl")
dict_wordVec_datapath = join(para.DATA_INPUT_DIRECTORY, "dict_wordVec.pkl")
dict_ontoee_event_rels_datapath = join(para.DATA_INPUT_DIRECTORY, "rel_dict_event-instance_pairs_data.pkl")
dict_ontoee_event_rels_supersub_datapath = join(para.DATA_INPUT_DIRECTORY, "rel_dict_super_sub_data.pkl")
dict_ontoee_train_all_datapath = join(para.DATA_INPUT_DIRECTORY, "Overall/event_dict_train_data.pkl")
dict_ontoee_valid_all_datapath = join(para.DATA_INPUT_DIRECTORY, "Overall/event_dict_valid_data.pkl")
dict_ontoee_test_all_datapath = join(para.DATA_INPUT_DIRECTORY, "Overall/event_dict_test_data.pkl")
dict_ontoee_train_low_datapath = join(para.DATA_INPUT_DIRECTORY, "Low-resource/event_dict_train_data.pkl")
dict_ontoee_valid_low_datapath = join(para.DATA_INPUT_DIRECTORY, "Low-resource/event_dict_valid_data.pkl")
dict_ontoee_test_low_datapath = join(para.DATA_INPUT_DIRECTORY, "Low-resource/event_dict_test_data.pkl")


max_length = para.MAX_LENGTH

if para.DATA_INPUT_TYPE == "Low":
    print("-----Training Data----")
    train_data_loader = DataLoder(dict_ontoee_train_low_datapath, dict_wordVec_datapath,
                                  dict_ontoee_event_rels_datapath, dict_ontoee_event_rels_supersub_datapath,
                                  model_name, max_length=max_length, pre_process=para.PRE_PROCESS, cuda=para.CUDA)
    print("-----Validating Data----")
    val_data_loader = DataLoder(dict_ontoee_valid_low_datapath, dict_wordVec_datapath,
                                dict_ontoee_event_rels_datapath, dict_ontoee_event_rels_supersub_datapath,
                                model_name, max_length=max_length, pre_process=para.PRE_PROCESS, cuda=para.CUDA)
    print("-----Testing Data----")
    test_data_loader = DataLoder(dict_ontoee_test_low_datapath, dict_wordVec_datapath,
                                 dict_ontoee_event_rels_datapath, dict_ontoee_event_rels_supersub_datapath,
                                 model_name, max_length=max_length, pre_process=para.PRE_PROCESS, cuda=para.CUDA)
    framework = LowResEDFramework(train_data_loader, val_data_loader, test_data_loader)
    sentence_encoder = BaselineSentenceEncoder(train_data_loader.word_vec_mat, 3, para.ENCODER_MODEL)
elif para.DATA_INPUT_TYPE == "All":
    print("-----Training Data----")
    train_data_loader = DataLoder(dict_ontoee_train_all_datapath, dict_wordVec_datapath,
                                  dict_ontoee_event_rels_datapath, dict_ontoee_event_rels_supersub_datapath,
                                  model_name, max_length=max_length, pre_process=para.PRE_PROCESS, cuda=para.CUDA)
    print("-----Validating Data----")
    val_data_loader = DataLoder(dict_ontoee_valid_all_datapath, dict_wordVec_datapath,
                                dict_ontoee_event_rels_datapath, dict_ontoee_event_rels_supersub_datapath,
                                model_name, max_length=max_length, pre_process=para.PRE_PROCESS, cuda=para.CUDA)
    print("-----Testing Data----")
    test_data_loader = DataLoder(dict_ontoee_test_all_datapath, dict_wordVec_datapath,
                                 dict_ontoee_event_rels_datapath, dict_ontoee_event_rels_supersub_datapath,
                                 model_name, max_length=max_length, pre_process=para.PRE_PROCESS, cuda=para.CUDA)
    framework = OverallEDFramework(train_data_loader, val_data_loader, test_data_loader)
    sentence_encoder = BaselineSentenceEncoder(train_data_loader.word_vec_mat, 3, para.ENCODER_MODEL)
else:
    raise NotImplementedError

if "-PPN" in model_name:
    framework = LowResEDFramework(train_data_loader, val_data_loader, test_data_loader)
    model = ProtoPropNet(sentence_encoder, sentence_encoder)
else:
    raise NotImplementedError

if para.DATA_INPUT_TYPE == "Low":
    framework.train(model, model_name, B, N_for_train, N_for_eval, R_for_train, R_for_eval, K, Q, noise_rate=noise_rate)
elif para.DATA_INPUT_TYPE == "All":
    framework.train(model, model_name, B, N_for_train, N_for_eval, 1, 1, 10000, 10000, noise_rate=noise_rate)
else:
    raise NotImplementedError
