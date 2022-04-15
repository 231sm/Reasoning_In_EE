# OntoED

🍎 This is an official implementation for [**OntoED**](https://github.com/231sm/Reasoning_In_EE/tree/main/OntoED) model, which has firstly been proposed in the paper [OntoED: Low-resource Event Detection with Ontology Embedding](https://arxiv.org/pdf/2105.10922.pdf) accepted by ACL 2021. 

## Code Structure
The structure of code and data is as follows: 

```
Reasoning_In_EE
 |-- OntoED  # model
 |    |-- data_utils.py  # for data processing
 |    |-- ontoed.py  # main model
 |    |-- run_ontoed.py  #  for model running
 |    |-- run_ontoed.sh  #  bash file for model running
 |-- OntoEvent  # data
 |    |-- event_dict_data_on_doc.json.zip   # raw full ED data
 |    |-- event_dict_train_data.json  #  ED data for training
 |    |-- event_dict_valid_data.json  #  ED data for validation
 |    |-- event_dict_test_data.json  #  ED data for testing
 |    |-- event_relation.json  #  event-event relation data
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

