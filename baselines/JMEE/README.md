# Jointly Multiple Events Extraction (JMEE)
This is the code of the Jointly Multiple Events Extraction (JMEE) in [EMNLP 2018 paper](https://arxiv.org/abs/1809.09078). ***Reproduced results with carefully hyper-parameter tuning and more training stpes, and we obtained better baseline results.***

## Requirement

- pytorch == 0.4.1.post2
- torchtext == 0.2.3
- tensorboardX == 2.5
- seqeval == 0.0.10
- numpy == 1.21.5

## Usage

To run this code, you need to:
- unzip the `standford.zip` dataset under `data` directory.
- prepare the GloVe embedding file `glove.6B.300d.txt` under `data` directory (you can download it [here](https://nlp.stanford.edu/projects/glove/)).
- create a `models` directory for checkpoint storage, if it doesn't exists.
- run the code using `train.sh`.

The hyper parameters are configured in `train.sh` and you may modify it as you wish.

## ED Performance & Reproduction

|       | Precision | Recall | F1-score |
| ----- | --------- | ------ | -------- |
| Macro | 50.52     | 53.75  | 49.87    |
| Micro | 72.26     | 72.26  | 72.26    |

You can download the trained checkpoint [here](https://drive.google.com/file/d/1oECaPmnOHtbXki6wldGbfrnK4IUo9RzC/view?usp=sharing), and run the `eval.sh` to reproduce the results.

