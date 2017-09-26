import argparse

from tensorpack.dataflow.remote import send_dataflow_zmq

from pose_dataset import get_dataflow, get_dataflow_batch

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training codes for Openpose using Tensorflow')
    parser.add_argument('--datapath', type=str, default='/data/public/rw/coco-pose-estimation-lmdb/')
    parser.add_argument('--batchsize', type=int, default=10)
    parser.add_argument('--train', type=bool, default=True)
    parser.add_argument('--master', type=str, default='tcp://brain-cluster-gpu8.dakao.io:1029')
    args = parser.parse_args()

    df = get_dataflow_batch(args.datapath, args.train, args.batchsize)

    send_dataflow_zmq(df, args.master)
