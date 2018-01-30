#!/bin/bash

echo "download model graph : cmu_640x480"

wget http://www.mediafire.com/file/q7fbue8dqr3ytrb/graph_freeze.pb -O graph_freeze.pb
wget http://www.mediafire.com/file/eolfk6t1t3yb191/graph_opt.pb -O graph_opt.pb
wget http://www.mediafire.com/file/s6d01qvmlkfxgzr/graph_q.pb -O graph_q.pb
wget http://www.mediafire.com/file/ae7hud583cx259z/graph.pb -O graph.pb

