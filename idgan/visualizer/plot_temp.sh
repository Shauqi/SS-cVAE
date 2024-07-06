#!/usr/bin/env bash

logger="plot_all.out"
echo "STARTING"

cherry_til23_idcs="1 2 3"

kwargs="-s 1234 -c 10 -r 20 -t 2 --is-show-loss --is-posterior"
python main_viz.py btcvae_til23 reconstruct-traverse -i $cherry_til23_idcs $kwargs