import argparse
import os
from shutil import copyfile

import logging
from tensorpack.dataflow.remote import send_dataflow_zmq

from pose_augment import set_network_input_wh
from pose_dataset import get_dataflow_batch


logging.basicConfig(level=logging.DEBUG, format='[lmdb_dataset] %(asctime)s %(levelname)s %(message)s')

if __name__ == '__main__':
    """
    OpenPose Data Preparation might be a bottleneck for training.
    You can run multiple workers to generate input batches in multi-nodes to make training process faster.
    """
    parser = argparse.ArgumentParser(description='Worker for preparing input batches.')
    parser.add_argument('--datapath', type=str, default='/data/public/rw/coco-pose-estimation-lmdb/')
    parser.add_argument('--batchsize', type=int, default=64)
    parser.add_argument('--train', type=bool, default=True)
    parser.add_argument('--copydb', type=bool, default=False)
    parser.add_argument('--master', type=str, default='tcp://csi-cluster-gpu20.dakao.io:1027')
    parser.add_argument('--input-width', type=int, default=368)
    parser.add_argument('--input-height', type=int, default=368)
    args = parser.parse_args()

    set_network_input_wh(args.input_width, args.input_height)

    if args.copydb:
        logging.info('db copy to local+')
        try:
            os.stat('/tmp/openposedb/')
        except:
            os.mkdir('/tmp/openposedb/')
        copyfile(args.datapath + 'data.mdb', '/tmp/openposedb/data.mdb')
        copyfile(args.datapath + 'lock.mdb', '/tmp/openposedb/lock.mdb')
        logging.info('db copy to local-')

        df = get_dataflow_batch('/tmp/openposedb/', args.train, args.batchsize)
    else:
        df = get_dataflow_batch(args.datapath, args.train, args.batchsize)

    send_dataflow_zmq(df, args.master, hwm=10)
