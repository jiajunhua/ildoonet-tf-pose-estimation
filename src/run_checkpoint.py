import argparse
import logging

import tensorflow as tf

from src.networks import get_network

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

config = tf.ConfigProto()
config.gpu_options.allocator_type = 'BFC'
config.gpu_options.per_process_gpu_memory_fraction = 0.95
config.gpu_options.allow_growth = True


if __name__ == '__main__':
    """
    inference with checkpoint file.
    While training, checkpoints are saved. You can test them with this python code.
    """
    parser = argparse.ArgumentParser(description='Tensorflow Pose Estimation Graph Extractor')
    parser.add_argument('--model', type=str, default='mobilenet', help='cmu / mobilenet / mobilenet_thin')
    args = parser.parse_args()

    input_node = tf.placeholder(tf.float32, shape=(1, 320, 480, 3), name='image')

    with tf.Session(config=config) as sess:
        net, _, last_layer = get_network(args.model, input_node, sess, trainable=False)

        tf.train.write_graph(sess.graph_def, '.', 'graph-tmp.pb', as_text=True)

        saver = tf.train.Saver(max_to_keep=100)
        saver.save(sess, '/Users/ildoonet/repos/tf-openpose/chk_tmp', global_step=1)
