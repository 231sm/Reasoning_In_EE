# OntoEvent

🍎 This is a repository for [**OntoEvent**](https://github.com/231sm/Reasoning_In_EE/tree/main/OntoEvent) dataset, which has firstly been proposed in the paper [OntoED: Low-resource Event Detection with Ontology Embedding](https://arxiv.org/pdf/2105.10922.pdf) accepted by ACL 2021. 

## Data File Structure
The structure of data files is as follows: 

```
Reasoning_In_EE
 |-- OntoEvent  # data
 |    |-- event_dict_data_on_doc.json.zip   # raw full ED data
 |    |-- event_dict_train_data.json  #  ED data for training
 |    |-- event_dict_valid_data.json  #  ED data for validation
 |    |-- event_dict_test_data.json  #  ED data for testing
 |    |-- event_relation.json  #  event-event relation data
```

## Brief Introduction
[**OntoEvent**](https://github.com/231sm/Reasoning_In_EE/tree/main/OntoEvent) is proposed for Event Detection and also annotated with correlations among events. It contains 13 supertypes with 100 subtypes, derived from 4,115 documents with 60,546 event instances. 

## Statistics
The statistics of OntoEvent are shown below, and the detailed data schema can be referred to our paper. 

Dataset         | #Doc | #Instance | #SuperType | #SubType | #EventCorrelation |
| :----------------- | ---------------- | ---------------- | ---------------- | ---------------- | ---------------- |
ACE 2005        | 599 | 4,090 | 8 | 33 | None |
TAC KBP 2017    | 167 | 4,839 | 8 | 18  | None |
FewEvent              | - | 70,852 | 19 | 100  | None |
MAVEN           | 4,480 | 111,611 | 21 | 168  | None |
$\texttt{OntoEvent}$    | 4,115 | 60,546 | 13 | 100 | 3,804 |

## Data Format
The OntoEvent dataset is stored in json format.

🍒For each *event instance* in [```event_dict_data_on_doc.json```](https://github.com/231sm/Reasoning_In_EE/blob/main/OntoEvent/event_dict_data_on_doc.json.zip)、[```event_dict_train_data.json```](https://github.com/231sm/Reasoning_In_EE/blob/main/OntoEvent/event_dict_train_data.json)、[```event_dict_valid_data.json```](https://github.com/231sm/Reasoning_In_EE/blob/main/OntoEvent/event_dict_valid_data.json)、[```event_dict_test_data.json```](https://github.com/231sm/Reasoning_In_EE/blob/main/OntoEvent/event_dict_test_data.json), the data format is as below:

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
