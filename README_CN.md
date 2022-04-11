[**中文**](https://github.com/231sm/Reasoning_In_EE/blob/main/README_CN.md) | [**English**](https://github.com/231sm/Reasoning_In_EE/blob/main/README.md)

<p align="center">
  	<font size=16><strong>OntoED：一个基于本体表示学习实现低资源事件检测的模型</strong></font>
</p>


🍎这个项目针对**OntoEvent**数据集和**OntoED**模型，是在[OntoED: Low-resource Event Detection with Ontology Embedding](https://arxiv.org/pdf/2105.10922.pdf)论文中提出的，该论文已被**ACL2021**主会录用。


# 项目简介
OntoED是解决低资源条件下事件检测的模型，通过本体表示学习建模了事件类别之间的关系：可以把样本多的事件类型知识迁移到样本少的事件类型上，而且未见过的事件类型可以通过本体建立起和已出现过的事件类型之间的联系。

# 关于数据集
**OntoEvent** 是针对事件检测任务提出的数据集，并且包含事件间关系的标注。其中包含13个父类，100个子类, 源于4,115 篇文档，涉及60,546 个事件实例。

## 数据统计
OntoEvent数据集的数据统计情况如下所示，其中详细的数据模式，比如事件上下位和事件间关联信息可以参考论文中的附录表格。

数据集 		| 文档数 | 实例数 | 父类数目 | 子类数目 | 事件间关联数目 |
| :----------------- | ---------------- | ---------------- | ---------------- | ---------------- | ---------------- |
ACE 2005 		| 599 | 4,090 | 8 | 33 | 无 |
TAC KBP 2017 	| 167 | 4,839 | 8 | 18  | 无 |
FewEvent 		      | - | 70,852 | 19 | 100  | 无 |
MAVEN 			| 4,480 | 111,611 | 21 | 168  | 无 |
OntoEvent	| 4,115 | 60,546 | 13 | 100 | 3,804 |

## 数据格式
OntoEvent数据集以json格式存储。

🍒对于```event_dict_data_on_doc.json```文件中的每个*事件实例* , 数据格式如下：

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

🍒对于```event_relation.json ```文件中的*事件关系* , 我们列出了*事件实例对*，数据格式如下：

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

🍒特别针对"COSUPER"、"SUBSUPER"、"SUPERSUB"这三个事件关系，我们列出了*事件类别对*，数据格式如下：

```
"COSUPER": [
    ["Conflict.Attack", "Conflict.Protest"], 
    ["Conflict.Attack", "Conflict.Sending"], 
    ...
]
```


# 有关项目
🤗 非常感谢您对我们的工作感兴趣。
我们很抱歉没有及时处理github上的提问，目前已上传完整原始版本的**OntoEvent**数据集。完整的项目代码还在公司服务器上，我们正在申请下载和披露，后面也将整理一下我们的代码，这需要一些时间。


# 有关论文
如果您使用或拓展我们的工作，请引用以下论文：

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



