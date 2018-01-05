import os

import tensorflow as tf
from network_hybrid_try import HybridNetworkTry
from network_mobilenet import MobilenetNetwork
from network_mobilenet_thin import MobilenetNetworkThin
from network_mobilenet_try import MobilenetNetworkTry
from network_mobilenet_try2 import MobilenetNetworkTry2
from network_mobilenet_try3 import MobilenetNetworkTry3

from src.network_cmu import CmuNetwork


def _get_base_path():
    if not os.environ.get('OPENPOSE_MODEL', ''):
        return './models'
    return os.environ.get('OPENPOSE_MODEL')


def get_network(type, placeholder_input, sess_for_load=None, trainable=True):
    if type == 'mobilenet':
        net = MobilenetNetwork({'image': placeholder_input}, conv_width=0.75, conv_width2=1.00, trainable=trainable)
        pretrain_path = 'pretrained/mobilenet_v1_0.75_224_2017_06_14/mobilenet_v1_0.75_224.ckpt'
        last_layer = 'MConv_Stage6_L{aux}_5'
    elif type == 'mobilenet_fast':
        net = MobilenetNetwork({'image': placeholder_input}, conv_width=0.75, conv_width2=0.50, trainable=trainable)
        pretrain_path = 'pretrained/mobilenet_v1_0.75_224_2017_06_14/mobilenet_v1_0.75_224.ckpt'
        last_layer = 'MConv_Stage6_L{aux}_5'
    elif type == 'mobilenet_thin':
        net = MobilenetNetworkThin({'image': placeholder_input}, conv_width=0.75, conv_width2=0.50, trainable=trainable)
        pretrain_path = 'pretrained/mobilenet_v1_0.75_224_2017_06_14/mobilenet_v1_1.0_224.ckpt'
        last_layer = 'MConv_Stage6_L{aux}_5'

    elif type == 'hybridnet_try':
        net = HybridNetworkTry({'image': placeholder_input}, conv_width2=1.00, trainable=trainable)
        pretrain_path = 'numpy/openpose_vgg16.npy'
        last_layer = 'MConv_Stage6_L{aux}_5'

    elif type == 'mobilenet_try':
        net = MobilenetNetworkTry({'image': placeholder_input}, conv_width=1.00, conv_width2=1.00, trainable=trainable)
        pretrain_path = 'pretrained/mobilenet_v1_1.0_224_2017_06_14/mobilenet_v1_1.0_224.ckpt'
        last_layer = 'MConv_Stage6_L{aux}_5'

    elif type == 'mobilenet_try2':
        net = MobilenetNetworkTry2({'image': placeholder_input}, conv_width=1.00, conv_width2=1.00, trainable=trainable)
        pretrain_path = 'pretrained/mobilenet_v1_1.0_224_2017_06_14/mobilenet_v1_1.0_224.ckpt'
        last_layer = 'MConv_Stage6_L{aux}_5'

    elif type == 'mobilenet_try3':
        net = MobilenetNetworkTry3({'image': placeholder_input}, conv_width=1.00, conv_width2=1.00, trainable=trainable)
        pretrain_path = 'pretrained/mobilenet_v1_1.0_224_2017_06_14/mobilenet_v1_1.0_224.ckpt'
        last_layer = 'MConv_Stage6_L{aux}_5'

    elif type == 'cmu':
        net = CmuNetwork({'image': placeholder_input}, trainable=trainable)
        pretrain_path = 'numpy/openpose_coco.npy'
        last_layer = 'Mconv7_stage6_L{aux}'
    elif type == 'vgg':
        net = CmuNetwork({'image': placeholder_input}, trainable=trainable)
        pretrain_path = 'numpy/openpose_vgg16.npy'
        last_layer = 'Mconv7_stage6_L{aux}'
    else:
        raise Exception('Invalid Mode.')

    if sess_for_load is not None:
        if type == 'cmu' or type == 'vgg':
            net.load(os.path.join(_get_base_path(), pretrain_path), sess_for_load)
        else:
            s = '%dx%d' % (placeholder_input.shape[2], placeholder_input.shape[1])
            ckpts = {
                'mobilenet': 'trained/mobilenet_%s/model-246038' % s,
                'mobilenet_thin': 'trained/mobilenet_thin_%s/model-449003' % s,
                'mobilenet_fast': 'trained/mobilenet_fast_%s/model-189000' % s,
                'mobilenet_accurate': 'trained/mobilenet_accurate/model-170000'
            }
            loader = tf.train.Saver()
            loader.restore(sess_for_load, os.path.join(_get_base_path(), ckpts[type]))

    return net, os.path.join(_get_base_path(), pretrain_path), last_layer


def get_graph_path(model_name):
    return {
        'cmu_320x240': './models/graph/cmu_320x240/graph_opt.pb',
        'cmu_480x320': './models/graph/cmu_480x320/graph_opt.pb',
        'cmu_640x480': './models/graph/cmu_640x480/graph_opt.pb',
        'mobilenet_320x240': './models/graph/mobilenet_320x240/graph_opt.pb',
        'mobilenet_fast_320x240': './models/graph/mobilenet_fast_320x240/graph_opt.pb',
        'mobilenet_thin_368x368': './models/trained/mobilenet_thin_368x368/graph_opt.pb'
    }[model_name]


def model_wh(model_name):
    width, height = model_name.split('_')[-1].split('x')
    return int(width), int(height)
