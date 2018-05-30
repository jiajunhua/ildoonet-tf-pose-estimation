import argparse
import logging
import sys
import time

import cv2
import numpy as np

from tf_pose import common
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

logger = logging.getLogger('TfPoseEstimator')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

Estimator = TfPoseEstimator

def infer(image, model='cmu', resize='0x0', resize_out_ratio=4.0):
    """

    :param image:
    :param model:
    :param resize:
    :param resize_out_ratio:
    :return:
    """
    w, h = model_wh(resize)
    if w == 0 or h == 0:
        e = TfPoseEstimator(get_graph_path(model), target_size=(432, 368))
    else:
        e = TfPoseEstimator(get_graph_path(model), target_size=(w, h))

    # estimate human poses from a single image !
    image = common.read_imgfile(image, None, None)
    if image is None:
        logger.error('Image can not be read, path=%s' % image)
        raise Exception('Image can not be read, path=%s' % image)
    humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=resize_out_ratio)

    return humans