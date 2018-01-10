#!/bin/bash

echo "download model graph : cmu_640x480"

wget https://www.dropbox.com/s/6za1f4zq5poat8b/graph_freeze.pb?dl=0 -O graph_freeze.pb
wget https://www.dropbox.com/s/l9t8rtwcyv6pzgt/graph_opt.pb?dl=0 -O graph_opt.pb
wget https://www.dropbox.com/s/x2nywsu7f2b7fv9/graph_q.pb?dl=0 -O graph_q.pb
wget https://www.dropbox.com/s/st5qp39r7jljvef/graph.pb?dl=0 -O graph.pb

