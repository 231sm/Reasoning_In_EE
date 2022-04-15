# DMCNN
Codes of the baseline model [DMCNN](https://www.aclweb.org/anthology/P15-1017/) for event detection.

## Requirements
- torch>=1.6
- numpy>=1.19.5
- sklearn==0.0
- seqeval==1.2.2
- tqdm==4.44.0

## Usage
To run this code, you need to:
1. unzip the `stanford.zip` dataset under the `data/` firectory, and `python convert.py` to convert data into MAVEN format as follows:
```JSON5
// Each `.jsonl` file is a subset of MAVEN and each line in the files is a json string for a document. For the `train.jsonl` and `valid.jsonl` the json format is as below:
{
    "id": "6b2e8c050e30872e49c2f46edb4ac044", // an unique string for each document
    "title": "Selma to Montgomery marches", // the tiltle of the document
    "content": [ // the content of the document. A list, each item is a dict for a sentence
    		{
    			"sentence": "...", // a string, the plain text of the sentence
    			"tokens": ["...", "..."] // a list, tokens of the sentence
		}
    ],
    "events":[ // a list for annotated events, each item is a dict for an event
        	{
            		"id": "75343904ec49aefe12c5749edadb7802", // an unique string for the event
            		"type": "Arranging", // the event type
            		"type_id": 70, // the numerical id for the event type
            		"mention":[ // a list for the event mentions of the event, each item is a dict
            			{
              				"id": "2db165c25298aefb682cba50c9327e4f", // an unique string for the event mention
              				"trigger_word": "organized", // a string of the trigger word or phrase
              				"sent_id": 1, // the index of the corresponding sentence, strates with 0
              				"offset": [3, 4], // the offset of the trigger words in the tokens list
              			}
             	     	]
        	},
    ],
    "negative_triggers":[ // a list for negative instances, each item is a dict for a negative mention
        {
        	"id": "46348f4078ae8460df4916d03573b7de",
            	"trigger_word": "desire",
            	"sent_id": 1,
            	"offset": [10, 11],
        },
    ]
}
```
2. Execute `run.sh` to perform training and testing. All the hyper-parameters are in config files at `./config/`, you can modify them as you wish.
