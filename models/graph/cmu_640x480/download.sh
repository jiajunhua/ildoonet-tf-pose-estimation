#!/usr/bin/env sh

echo "download model graph : cmu_640x480"

extract_download_url() {

        url=$( wget -q -O - $1 |  grep -o 'http*://download[^"]*' | tail -n 1 )
        echo "$url"

}

wget $( extract_download_url http://www.mediafire.com/file/q7fbue8dqr3ytrb/graph_freeze.pb ) -O graph_freeze.pb
wget $( extract_download_url http://www.mediafire.com/file/eolfk6t1t3yb191/graph_opt.pb ) -O graph_opt.pb
wget $( extract_download_url http://www.mediafire.com/file/s6d01qvmlkfxgzr/graph_q.pb ) -O graph_q.pb
wget $( extract_download_url http://www.mediafire.com/file/ae7hud583cx259z/graph.pb ) -O graph.pb
