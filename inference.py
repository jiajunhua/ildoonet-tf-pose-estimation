import pickle
import tensorflow as tf
import cv2
import numpy as np
import time
import logging
import argparse

from tensorflow.python.client import timeline

from network_cmu import CmuNetwork
from common import estimate_pose, CocoPairsRender, read_imgfile, CocoColors
from network_dsconv import DSConvNetwork
from network_mobilenet import MobilenetNetwork
from pose_dataset import CocoPoseLMDB

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

config = tf.ConfigProto()
config.gpu_options.allocator_type = 'BFC'
config.gpu_options.per_process_gpu_memory_fraction = 0.95
config.gpu_options.allow_growth = True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tensorflow Openpose Inference')
    parser.add_argument('--imgpath', type=str, default='./images/person3.jpg')
    parser.add_argument('--input-width', type=int, default=368)
    parser.add_argument('--input-height', type=int, default=368)
    parser.add_argument('--stage-level', type=int, default=6)
    parser.add_argument('--model', type=str, default='mobilenet', help='cmu / mobilenet / mobilenet_accurate / mobilenet_fast')
    args = parser.parse_args()

    input_node = tf.placeholder(tf.float32, shape=(1, args.input_height, args.input_width, 3), name='image')

    with tf.Session(config=config) as sess:
        if args.model == 'cmu':
            net = CmuNetwork({'image': input_node}, trainable=False)
            net.load('./models/numpy/openpose_coco.npy', sess)
            last_layer = 'Mconv7_stage{stage}_L{aux}'
        elif args.model == 'mobilenet_accurate':
            net = MobilenetNetwork({'image': input_node}, trainable=False, conv_width=1.0)
            loader = tf.train.Saver()
            loader.restore(sess, './models/trained/mobilenet_accurate/model-170000')
            last_layer = 'MConv_Stage{stage}_L{aux}_5'
        elif args.model == 'mobilenet_fast':
            net = MobilenetNetwork({'image': input_node}, trainable=False, conv_width=0.5)
            loader = tf.train.Saver()
            loader.restore(sess, './models/trained/mobilenet_fast/model-163000')
            last_layer = 'MConv_Stage{stage}_L{aux}_5'
        elif args.model == 'mobilenet':
            net = MobilenetNetwork({'image': input_node}, trainable=False, conv_width=0.75, conv_width2=0.50)
            loader = tf.train.Saver()
            loader.restore(sess, './models/trained/mobilenet/model-241003')
            last_layer = 'MConv_Stage{stage}_L{aux}_5'
        else:
            raise Exception('Invalid Mode.')

        logging.debug('read image+')
        image = read_imgfile(args.imgpath, args.input_width, args.input_height)
        vec = sess.run(net.get_output(name='concat_stage7'), feed_dict={'image:0': [image]})

        a = time.time()
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        pafMat, heatMat = sess.run(
            [
                net.get_output(name=last_layer.format(stage=args.stage_level, aux=1)),
                net.get_output(name=last_layer.format(stage=args.stage_level, aux=2))
            ], feed_dict={'image:0': [image]}, options=run_options, run_metadata=run_metadata
        )
        logging.info('inference- elapsed_time={}'.format(time.time() - a))

        tl = timeline.Timeline(run_metadata.step_stats)
        ctf = tl.generate_chrome_trace_format()
        with open('timeline.json', 'w') as f:
            f.write(ctf)
        heatMat, pafMat = heatMat[0], pafMat[0]

        logging.debug('inference+')

        avg = 0
        for _ in range(10):
            a = time.time()
            sess.run(
                [
                    net.get_output(name=last_layer.format(stage=args.stage_level, aux=1)),
                    net.get_output(name=last_layer.format(stage=args.stage_level, aux=2))
                ], feed_dict={'image:0': [image]}
            )
            logging.info('inference- elapsed_time={}'.format(time.time() - a))
            avg += time.time() - a
        logging.info('prediction avg= %f' % (avg / 10))

        '''
        logging.info('pickle data')
        with open('person3.pickle', 'wb') as pickle_file:
            pickle.dump(image, pickle_file, pickle.HIGHEST_PROTOCOL)
        with open('heatmat.pickle', 'wb') as pickle_file:
            pickle.dump(heatMat, pickle_file, pickle.HIGHEST_PROTOCOL)
        with open('pafmat.pickle', 'wb') as pickle_file:
            pickle.dump(pafMat, pickle_file, pickle.HIGHEST_PROTOCOL)
        '''

        logging.info('pose+')
        a = time.time()
        humans = estimate_pose(heatMat, pafMat)
        logging.info('pose- elapsed_time={}'.format(time.time() - a))

        logging.info('image={} heatMap={} pafMat={}'.format(image.shape, heatMat.shape, pafMat.shape))
        process_img = CocoPoseLMDB.display_image(image, heatMat, pafMat, as_numpy=True)

        # display
        image = cv2.imread(args.imgpath)
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
                image = cv2.line(image, center1, center2, CocoColors[part_idx], 1)

        scale = 480.0 / image_h
        newh, neww = 480, int(scale * image_w + 0.5)

        image = cv2.resize(image, (neww, newh), interpolation=cv2.INTER_AREA)

        convas = np.zeros([480, 640 + neww, 3], dtype=np.uint8)
        convas[:, :640] = process_img
        convas[:, 640:] = image

        cv2.imshow('result', convas)
        cv2.waitKey(0)

        tf.train.write_graph(sess.graph_def, '.', 'graph-tmp.pb', as_text=True)
