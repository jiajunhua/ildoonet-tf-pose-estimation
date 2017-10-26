import os

import tensorflow as tf

from network_cmu import CmuNetwork
from network_mobilenet import MobilenetNetwork


def _get_base_path():
    if not os.environ.get('OPENPOSE_MODEL', ''):
        return './models'
    return os.environ.get('OPENPOSE_MODEL')


def get_network(type, placeholder_input, sess_for_load=None):
    if type == 'mobilenet':
        net = MobilenetNetwork({'image': placeholder_input}, conv_width=0.75, conv_width2=0.50)
        pretrain_path = 'pretrained/mobilenet_v1_0.75_224_2017_06_14/mobilenet_v1_0.75_224.ckpt'
        last_layer = 'MConv_Stage6_L{aux}_5'
    elif type == 'mobilenet_accurate':
        net = MobilenetNetwork({'image': placeholder_input}, conv_width=1.00)
        pretrain_path = 'pretrained/mobilenet_v1_1.0_224_2017_06_14/mobilenet_v1_1.0_224.ckpt'
        last_layer = 'MConv_Stage6_L{aux}_5'
    elif type == 'mobilenet_fast':
        net = MobilenetNetwork({'image': placeholder_input}, conv_width=0.50)
        pretrain_path = 'pretrained/mobilenet_v1_0.50_224_2017_06_14/mobilenet_v1_0.50_224.ckpt'
        last_layer = 'MConv_Stage6_L{aux}_5'
    elif type == 'cmu':
        net = CmuNetwork({'image': placeholder_input})
        pretrain_path = 'numpy/openpose_coco.npy'
        last_layer = 'Mconv7_stage6_L{aux}'
    else:
        raise Exception('Invalid Mode.')

    if sess_for_load is not None:
        if type == 'cmu':
            net.load('./models/numpy/openpose_coco.npy', sess_for_load)
        else:
            s = '%dx%d' % (placeholder_input.shape[2], placeholder_input.shape[1])
            ckpts = {
                'mobilenet': 'trained/mobilenet_%s/model-release' % s,
                'mobilenet_fast': 'trained/mobilenet_fast/model-163000',
                'mobilenet_accurate': 'trained/mobilenet_accurate/model-170000'
            }
            loader = tf.train.Saver()
            loader.restore(sess_for_load, os.path.join(_get_base_path(), ckpts[type]))

    return net, os.path.join(_get_base_path(), pretrain_path), last_layer
