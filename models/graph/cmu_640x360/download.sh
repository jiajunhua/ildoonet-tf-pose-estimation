#!/bin/bash

echo "download model graph : cmu_640x360"

wget http://download1662.mediafire.com/9w41h1qg57og/fdhqpmdrdzafoc4/graph_freeze.pb -O graph_freeze.pb
wget http://download1650.mediafire.com/35hcd7ukp3fg/n6qnqz00g1pjf7d/graph_opt.pb -O graph_opt.pb
wget http://download1193.mediafire.com/eaoeszlwevfg/38hyjrwfdsyqsbq/graph_q.pb -O graph_q.pb
wget http://download1477.mediafire.com/5mujvsj810xg/a2a0nc8i1oj5iam/graph.pb -O graph.pb
