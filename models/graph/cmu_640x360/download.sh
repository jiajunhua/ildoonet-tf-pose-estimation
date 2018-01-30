#!/bin/bash

echo "download model graph : cmu_640x360"

wget http://www.mediafire.com/file/fdhqpmdrdzafoc4/graph_freeze.pb -O graph_freeze.pb
wget http://www.mediafire.com/file/n6qnqz00g1pjf7d/graph_opt.pb -O graph_opt.pb
wget http://www.mediafire.com/file/38hyjrwfdsyqsbq/graph_q.pb -O graph_q.pb
wget http://www.mediafire.com/file/a2a0nc8i1oj5iam/graph.pb -O graph.pb
