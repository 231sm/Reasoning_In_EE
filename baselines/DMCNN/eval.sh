#!/usr/bin/env bash

MODEL_PATH="ckpt/dmcnn.pt"  # Configure path to checkpoint file of pre-trained model

rm -rf data/Maven/
python main.py --config dmcnn.config --gpu 0 --test_only --ckpt $MODEL_PATH
