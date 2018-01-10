#!/bin/bash

echo "download model graph : cmu_640x360"

wget https://www.dropbox.com/s/i442ol5ig3tn0h8/graph_freeze.pb?dl=0 -O graph_freeze.pb
wget https://www.dropbox.com/s/ppo3b44jdfoz94x/graph_opt.pb?dl=0 -O graph_opt.pb
wget https://www.dropbox.com/s/fvqx4r6idix15mn/graph_q.pb?dl=0 -O graph_q.pb
wget https://www.dropbox.com/s/u7g1e4i1fc377rg/graph.pb?dl=0 -O graph.pb
