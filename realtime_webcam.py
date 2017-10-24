import argparse
import cv2
import numpy as np
import time
import logging

import tensorflow as tf

from common import CocoPairsRender, CocoColors, preprocess, estimate_pose
from network_cmu import CmuNetwork
from network_mobilenet import MobilenetNetwork
from pose_dataset import CocoPoseLMDB

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')


fps_time = 0


def cb_showimg(img, preprocessed, heatMat, pafMat, humans, show_process=False):
    global fps_time

    # display
    image = img
    image_h, image_w = image.shape[:2]
    heat_h, heat_w = heatMat.shape[:2]
    for _, human in humans.items():
        for part_idx, part in enumerate(human):
            if part['partIdx'] not in CocoPairsRender:
                continue
            center1 = (int((part['c1'][0] + 0.5) * image_w / heat_w), int((part['c1'][1] + 0.5) * image_h / heat_h))
            center2 = (int((part['c2'][0] + 0.5) * image_w / heat_w), int((part['c2'][1] + 0.5) * image_h / heat_h))
            cv2.circle(image, center1, 3, part['partIdx'][0], thickness=3, lineType=8, shift=0)
            cv2.circle(image, center2, 3, part['partIdx'][1], thickness=3, lineType=8, shift=0)
            image = cv2.line(image, center1, center2, CocoColors[part_idx % len(CocoColors)], 1)

    scale = 480.0 / image_h
    newh, neww = 480, int(scale * image_w + 0.5)

    image = cv2.resize(image, (neww, newh), interpolation=cv2.INTER_AREA)

    if show_process:
        process_img = CocoPoseLMDB.display_image(preprocessed, heatMat, pafMat, as_numpy=True)

        canvas = np.zeros([480, 640 + neww, 3], dtype=np.uint8)
        # canvas[:, :640] = process_img
        canvas[:, 640:] = image
    else:
        canvas = image

    cv2.putText(canvas, "FPS: %f" % (1.0 / (time.time() - fps_time)), (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.imshow('openpose', canvas)

    fps_time = time.time()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tensorflow Openpose Realtime Webcam')
    parser.add_argument('--input-width', type=int, default=368)
    parser.add_argument('--input-height', type=int, default=368)
    parser.add_argument('--stage-level', type=int, default=6)
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--model', type=str, default='mobilenet_fast', help='cmu / mobilenet / mobilenet_accurate / mobilenet_fast')
    parser.add_argument('--show-process', type=bool, default=False, help='for debug purpose, if enabled, speed for inference is dropped.')
    args = parser.parse_args()

    cam = cv2.VideoCapture(args.camera)

    input_node = tf.placeholder(tf.float32, shape=(1, args.input_height, args.input_width, 3), name='image')

    with tf.Session() as sess:
        if args.model == 'cmu':
            net = CmuNetwork({'image': input_node}, trainable=False)
            net.load('./models/numpy/openpose_coco.npy', sess)
            last_layer = 'Mconv7_stage{stage}_L{aux}'
        elif args.model == 'mobilenet_accurate':
            net = MobilenetNetwork({'image': input_node}, trainable=False, conv_width=1.0)
            loader = tf.train.Saver()
            loader.restore(sess, '/Users/ildoonet/Downloads/best_mobilenet_p_1.0/model-26000')
            last_layer = 'MConv_Stage{stage}_L{aux}_5'
        elif args.model == 'mobilenet_fast':
            net = MobilenetNetwork({'image': input_node}, trainable=False, conv_width=0.5)
            loader = tf.train.Saver()
            # loader.restore(sess, '/Users/ildoonet/Downloads/openpose-mobilenet_0.50/model-67000')
            loader.restore(sess, '/Users/ildoonet/Downloads/openpose-mobilenet_0.50/model-163000')
            last_layer = 'MConv_Stage{stage}_L{aux}_5'
        elif args.model == 'mobilenet':
            net = MobilenetNetwork({'image': input_node}, trainable=False, conv_width=0.75, conv_width2=0.50)
            loader = tf.train.Saver()
            loader.restore(sess, '/Users/ildoonet/Downloads/openpose-mobilenet_0.75_0.50/model-217003')
            last_layer = 'MConv_Stage{stage}_L{aux}_5'
        else:
            raise Exception('Invalid Mode.')

        while True:
            logging.debug('cam read+')
            ret_val, img = cam.read()

            logging.debug('cam preprocess+')
            preprocessed = preprocess(img, args.input_width, args.input_height)

            logging.debug('cam process+')
            pafMat, heatMat = sess.run(
                [
                    net.get_output(name=last_layer.format(stage=args.stage_level, aux=1)),
                    net.get_output(name=last_layer.format(stage=args.stage_level, aux=2))
                ], feed_dict={'image:0': [preprocessed]}
            )
            heatMat, pafMat = heatMat[0], pafMat[0]

            logging.debug('cam postprocess+')
            t = time.time()
            humans = estimate_pose(heatMat, pafMat)

            logging.debug('cam show+')
            cb_showimg(img, preprocessed, heatMat, pafMat, humans, show_process=args.show_process)

            if cv2.waitKey(1) == 27:
                break  # esc to quit
            logging.debug('cam finished+')
    cv2.destroyAllWindows()
