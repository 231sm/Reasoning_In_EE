[**中文**](https://github.com/231sm/Reasoning_In_EE/blob/main/README_CN.md) | [**English**](https://github.com/231sm/Reasoning_In_EE/blob/main/README.md)

<p align="center">
  	<font size=50><strong>OntoED: A Model for Low-resource Event Detection with Ontology Embedding</strong></font>
</p>


🍎This repository is a official repository for **OntoEvent** dataset and **OntoED**  model, which has firstly proposed in a paper: [OntoED: Low-resource Event Detection with Ontology Embedding](https://arxiv.org/pdf/2105.10922.pdf), accepted by ACL 2021 main conference. 

# Brief Introduction
OntoED is a model that resolves event detection under low-resource conditions. It models the relationship between event types through ontology embedding: it can transfer knowledge of high-resource event types to low-resource ones, and the unseen event type can establish connection with seen ones via event ontology.

# How about the Dataset
**OntoEvent**  is proposed for ED and also annotated with correlations among events. It contains 13 supertypes with 100 subtypes, derived from 4,115 documents with 60,546 event instances. 

## Statistics
The statistics of OntoEvent are shown below, and the detailed data schema can be referred to our paper. 

Dataset 		| #Doc | #Instance | #SuperType | #SubType | #EventCorrelation |
| :----------------- | ---------------- | ---------------- | ---------------- | ---------------- | ---------------- |
ACE 2005 		| 599 | 4,090 | 8 | 33 | None |
TAC KBP 2017 	| 167 | 4,839 | 8 | 18  | None |
FewEvent 		      | - | 70,852 | 19 | 100  | None |
MAVEN 			| 4,480 | 111,611 | 21 | 168  | None |
OntoEvent	| 4,115 | 60,546 | 13 | 100 | 3,804 |

## Data Format
The OntoEvent dataset is stored in json format.

🍒For each *event instance* in ```event_dict_data_on_doc.json```, the data format is as below:

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
🍒For each *event relation* in ```event_relation.json```, we list the *event instance pair*, and the data format is as below:

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



# How about the Project
🤗 Thank you very much for your interest in our work. 
We are sorry for not dealing with the issues in time, and have uploaded full original version of **OntoEvent** dataset. The full project with code is still on the company server, and we are applying for downloading as well as disclosure, further remanaging the code, which would cost a few days. 


# Papers for the Project & How to Cite
📋 If you use or extend our work, please cite the following paper:

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
