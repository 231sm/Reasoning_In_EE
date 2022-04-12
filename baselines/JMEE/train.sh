#!/usr/bin/env bash

python -m enet.run.ee.runner \
 --device "cuda:0" \
 --train "data/stanford/train_form.json" \
 --test "data/stanford/test_form.json" \
 --dev "data/stanford/valid_form.json" \
 --webd "data/glove.6B.300d.txt" \
 --earlystop 10 --restart 999999 \
 --optimizer "adam" \
 --lr 2e-2 \
 --epochs 10 \
 --batch 128 \
 --out "models/enet" \
 --hps "{'wemb_dim': 300, 'wemb_ft': True, 'wemb_dp': 0.5, 'pemb_dim': 50, 'pemb_dp': 0.5, 'eemb_dim': 50, 'eemb_dp': 0.5, 'psemb_dim': 50, 'psemb_dp': 0.5, 'lstm_dim': 200, 'lstm_layers': 1, 'lstm_dp': 0, 'gcn_et': 3, 'gcn_use_bn': True, 'gcn_layers': 3, 'gcn_dp': 0.3, 'sa_dim': 200, 'use_highway': True, 'loss_alpha': 1.0}" \
#  >& enet.log &
