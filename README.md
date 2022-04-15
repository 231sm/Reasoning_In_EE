# OntoED and OntoEvent

<p align="center">
    <font size=4><strong>OntoED: A Model for Low-resource Event Detection with Ontology Embedding</strong></font>
</p>


🍎 The project is an official implementation for [**OntoED**](https://github.com/231sm/Reasoning_In_EE/tree/main/OntoED) model and a repository for [**OntoEvent**](https://github.com/231sm/Reasoning_In_EE/tree/main/OntoEvent) dataset, which has firstly been proposed in the paper [OntoED: Low-resource Event Detection with Ontology Embedding](https://arxiv.org/pdf/2105.10922.pdf) accepted by ACL 2021. 

🤗 The implementations are based on [Huggingface's Transformers](https://github.com/huggingface/transformers) and remanagement is referred to [MAVEN's baselines](https://github.com/THU-KEG/MAVEN-dataset/) & [DeepKE](https://github.com/zjunlp/DeepKE). 

🤗 We also provide some [baseline implementations](https://github.com/231sm/Reasoning_In_EE/tree/main/baselines) for reproduction. 


## Brief Introduction
OntoED is a model that resolves event detection under low-resource conditions. It models the relationship between event types through ontology embedding: it can transfer knowledge of high-resource event types to low-resource ones, and the unseen event type can establish connection with seen ones via event ontology.


## Project Structure
The structure of data and code is as follows: 

```
Reasoning_In_EE
 |-- OntoEvent  # data
 |    |-- event_dict_data_on_doc.json.zip   # raw full ED data
 |    |-- event_dict_train_data.json  #  ED data for training
 |    |-- event_dict_valid_data.json  #  ED data for validation
 |    |-- event_dict_test_data.json  #  ED data for testing
 |    |-- event_relation.json  #  event-event relation data
 |-- OntoED  # model
 |    |-- data_utils.py  # for data processing
 |    |-- ontoed.py  # main model
 |    |-- run_ontoed.py  #  for model running
 |    |-- run_ontoed.sh  #  bash file for model running
 |-- baselines # baseline models
 |    |-- DMCNN
 |    |    |-- will complete file information
 |    |-- JMEE
 |    |    |-- will complete file information
 |    |-- JRNN
 |    |    |-- will complete file information
```

## Requirements

- python==3.6.9

- torch==1.8.0 (lower may also be OK)

- transformers==2.8.0

- sklearn==0.20.2

- seqeval


## Usage


**1. Project Preparation**：Download this project and unzip the dataset. You can directly download the archive, or run ```git clone https://github.com/231sm/Reasoning_In_EE.git``` at your teminal. 

```
cd [LOCAL_PROJECT_PATH]

git clone https://github.com/231sm/Reasoning_In_EE.git
```

**2. Running Preparation**: Adjust the parameters in [```run_ontoed.sh```](https://github.com/231sm/Reasoning_In_EE/tree/main/OntoED/run_ontoed.sh) bash file, and input the true path of 'LABEL\_PATH' and 'RELATION\_PATH' at the end of [```data_utils.py```](https://github.com/231sm/Reasoning_In_EE/tree/main/OntoED/data_utils.py). 

```
cd Reasoning_In_EE/OntoED

vim run_ontoed.sh
(input the parameters, save and quit)

vim data_utils.py
(input the path of 'LABEL_PATH' and 'RELATION_PATH', save and quit)
```
**Hint**:  

- Please refer to ```main()``` function in [```run_ontoed.py```](https://github.com/231sm/Reasoning_In_EE/tree/main/OntoED/run_ontoed.py) file for detail meanings of each parameters. 
- 'LABEL\_PATH' and 'RELATION\_PATH' means the path for [event\_dict\_train_data.json](https://github.com/231sm/Reasoning_In_EE/tree/main/OntoEvent/event_dict_train_data.json) and [event_relation.json](https://github.com/231sm/Reasoning_In_EE/tree/main/OntoEvent/event_relation.json) respectively. 

**3. Running Model**: Run [```./run_ontoed.sh```](https://github.com/231sm/Reasoning_In_EE/tree/main/OntoED/run_ontoed.sh) for *training*, *validation*, and *testing*. 
A folder with configuration, models weights, and results (in ```is_test_true_eval_results.txt```) will be saved at the path you input ('--output_dir') in the bash file [```run_ontoed.sh```](https://github.com/231sm/Reasoning_In_EE/tree/main/OntoED/run_ontoed.sh). 

```
cd Reasoning_In_EE/OntoED

./run_ontoed.sh
('--do_train', '--do_eval', '--evaluate_during_training', '--test' is necessarily input in 'run_ontoed.sh')

Or you can run run_ontoed.py with manual parameter input (parameters can be copied from 'run_ontoed.sh')

python run_ontoed.py --para... 
```


## How about the Dataset
[**OntoEvent**](https://github.com/231sm/Reasoning_In_EE/tree/main/OntoEvent)  is proposed for ED and also annotated with correlations among events. It contains 13 supertypes with 100 subtypes, derived from 4,115 documents with 60,546 event instances. 
Please refer to [OntoEvent](https://github.com/231sm/Reasoning_In_EE/tree/main/OntoEvent) for details. 

### Statistics
The statistics of OntoEvent are shown below, and the detailed data schema can be referred to our paper. 

Dataset         | #Doc | #Instance | #SuperType | #SubType | #EventCorrelation |
| :----------------- | ---------------- | ---------------- | ---------------- | ---------------- | ---------------- |
ACE 2005        | 599 | 4,090 | 8 | 33 | None |
TAC KBP 2017    | 167 | 4,839 | 8 | 18  | None |
FewEvent              | - | 70,852 | 19 | 100  | None |
MAVEN           | 4,480 | 111,611 | 21 | 168  | None |
$\texttt{OntoEvent}$    | 4,115 | 60,546 | 13 | 100 | 3,804 |

### Data Format
The OntoEvent dataset is stored in json format.

🍒For each *event instance* in [```event_dict_data_on_doc.json```](https://github.com/231sm/Reasoning_In_EE/blob/main/OntoEvent/event_dict_data_on_doc.json.zip), the data format is as below:

```
{
    'doc_id': '...', 
    'doc_title': 'XXX', 
    'sent_id': , 
    'event_mention': '......', 
    'event_mention_tokens': ['.', '.', '.', '.', '.', '.'], 
    'trigger': '...', 
    'trigger_pos': [, ], 
    'event_type': ''
}
```
🍒For each *event relation* in [```event_relation.json```](https://github.com/231sm/Reasoning_In_EE/blob/main/OntoEvent/event_relation.json), we list the *event instance pair*, and the data format is as below:

```
'EVENT_RELATION_1': [ 
    [
        {
            'doc_id': '...', 
            'doc_title': 'XXX', 
            'sent_id': , 
            'event_mention': '......', 
            'event_mention_tokens': ['.', '.', '.', '.', '.', '.'], 
            'trigger': '...', 
            'trigger_pos': [, ], 
            'event_type': ''
        }, 
        {
            'doc_id': '...', 
            'doc_title': 'XXX', 
            'sent_id': , 
            'event_mention': '......', 
            'event_mention_tokens': ['.', '.', '.', '.', '.', '.'], 
            'trigger': '...', 
            'trigger_pos': [, ], 
            'event_type': ''
        }
    ], 
    ...
]
```
🍒Especially for "COSUPER", "SUBSUPER" and "SUPERSUB", we list the *event type pair*, and the data format is as below:

```
"COSUPER": [
    ["Conflict.Attack", "Conflict.Protest"], 
    ["Conflict.Attack", "Conflict.Sending"], 
    ...
]
```


## How to Cite
📋 Thank you very much for your interest in our work. If you use or extend our work, please cite the following paper:

```
@inproceedings{ACL2021_OntoED,
    title = "{O}nto{ED}: Low-resource Event Detection with Ontology Embedding",
    author = "Deng, Shumin  and
      Zhang, Ningyu  and
      Li, Luoqiu  and
      Hui, Chen  and
      Huaixiao, Tou  and
      Chen, Mosha  and
      Huang, Fei  and
      Chen, Huajun",
    booktitle = "Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.acl-long.220",
    doi = "10.18653/v1/2021.acl-long.220",
    pages = "2828--2839"
}
```
