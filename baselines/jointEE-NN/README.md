# Joint Event Extraction via Recurrent Neural Networks

This is the code for Joint Event Extraction RNN (JRNN) in [NAACL 2016 paper](https://aclanthology.org/N16-1034.pdf).

## Requirements

- python2
- theano

## Usage

To run this code, you need to:

* Preprocessing: using file ```jee_processData.py```

You will need to have the ACE 2005 data set in the format required by this file.
We cannot include the data in this release due to licence issues. A sample document is provided in ```data```.

* Train and test the model: using file ```evaluateJEE.py```

This step takes the output file in step 1.

There are various parameters of the model you can change directly in the ```evaluateJEE.py``` file.
