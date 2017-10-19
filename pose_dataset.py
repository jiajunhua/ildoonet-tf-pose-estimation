import sys
import math
import struct
import threading
import logging
import multiprocessing

from contextlib import contextmanager

import lmdb
import cv2
import numpy as np
import time

import tensorflow as tf

from tensorpack import imgaug
from tensorpack.dataflow.image import MapDataComponent, AugmentImageComponent
from tensorpack.dataflow.common import BatchData, MapData, TestDataSpeed
from tensorpack.dataflow.prefetch import PrefetchData
from tensorpack.dataflow.base import RNGDataFlow, DataFlowTerminated

from datum_pb2 import Datum
from pose_augment import pose_flip, pose_rotation, pose_to_img, pose_crop_random, \
    pose_resize_shortestedge_random, pose_resize_shortestedge_fixed, pose_crop_center, pose_random_scale

import matplotlib as mpl

logging.basicConfig(level=logging.DEBUG, format='[lmdb_dataset] %(asctime)s %(levelname)s %(message)s')


class CocoMetadata:
    # __coco_parts = 57
    __coco_parts = 19
    __coco_vecs = list(zip(
        [2, 9,  10, 2,  12, 13, 2, 3, 4, 3,  2, 6, 7, 6,  2, 1,  1,  15, 16],
        [9, 10, 11, 12, 13, 14, 3, 4, 5, 17, 6, 7, 8, 18, 1, 15, 16, 17, 18]
    ))

    @staticmethod
    def parse_float(four_np):
        assert len(four_np) == 4
        return struct.unpack('<f', bytes(four_np))[0]

    @staticmethod
    def parse_floats(four_nps, adjust=0):
        assert len(four_nps) % 4 == 0
        return [(CocoMetadata.parse_float(four_nps[x*4:x*4+4]) + adjust) for x in range(len(four_nps) // 4)]

    def __init__(self, idx, img, meta, sigma):
        self.idx = idx
        self.img = img
        self.sigma = sigma

        self.height = int(CocoMetadata.parse_float(meta[1][:4]))
        self.width = int(CocoMetadata.parse_float(meta[1][4:8]))

        self.num_other_people = meta[2][1]
        self.people_index = meta[2][2]

        # self.objpos_x = CocoMetadata.parse_float(meta[3][:4]) - 1
        # self.objpos_y = CocoMetadata.parse_float(meta[3][4:8]) - 1

        # self.objpos = [(self.objpos_x, self.objpos_y)]

        joint_list = []
        joint_x = CocoMetadata.parse_floats(meta[5][:CocoMetadata.__coco_parts*4], adjust=-1)
        joint_y = CocoMetadata.parse_floats(meta[6][:CocoMetadata.__coco_parts*4], adjust=-1)
        joint_list.append(list(zip(joint_x, joint_y)))

        for person_idx in range(self.num_other_people):
            # objpos_x = CocoMetadata.parse_float(meta[8+person_idx][:4]) - 1
            # objpos_y = CocoMetadata.parse_float(meta[8+person_idx][4:8]) - 1
            # self.objpos.append((objpos_x, objpos_y))

            joint_x = CocoMetadata.parse_floats(meta[9+self.num_other_people+3*person_idx][:CocoMetadata.__coco_parts*4], adjust=-1)
            joint_y = CocoMetadata.parse_floats(meta[9+self.num_other_people+3*person_idx+1][:CocoMetadata.__coco_parts*4], adjust=-1)
            joint_x = [val for val in joint_x if val >= 0 or -1000]
            joint_y = [val for val in joint_y if val >= 0 or -1000]
            joint_list.append(list(zip(joint_x, joint_y)))

        self.joint_list = []
        transform = list(zip(
            [1, 6, 7, 9, 11, 6, 8, 10, 13, 15, 17, 12, 14, 16, 3, 2, 5, 4],
            [1, 7, 7, 9, 11, 6, 8, 10, 13, 15, 17, 12, 14, 16, 3, 2, 5, 4]
        ))
        for prev_joint in joint_list:
            new_joint = []
            for idx1, idx2 in transform:
                j1 = prev_joint[idx1-1]
                j2 = prev_joint[idx2-1]

                if j1[0] <= 0 or j1[1] <= 0 or j2[0] <= 0 or j2[1] <= 0:
                    new_joint.append((-1000, -1000))
                else:
                    new_joint.append(((j1[0] + j2[0]) / 2, (j1[1] + j2[1]) / 2))

            new_joint.append((-1000, -1000))
            self.joint_list.append(new_joint)

        logging.debug('joint size=%d' % len(self.joint_list))

    def get_heatmap(self, target_size):
        heatmap = np.zeros((CocoMetadata.__coco_parts, self.height, self.width))

        for joints in self.joint_list:
            for idx, point in enumerate(joints):
                if point[0] < 0 or point[1] < 0:
                    continue
                CocoMetadata.put_heatmap(heatmap, idx, point, self.sigma)

        heatmap = heatmap.transpose((1, 2, 0))

        # background
        heatmap[:, :, -1] = np.clip(1 - np.amax(heatmap, axis=2), 0.0, 1.0)

        if target_size:
            heatmap = cv2.resize(heatmap, target_size, interpolation=cv2.INTER_AREA)

        return heatmap

    @staticmethod
    def put_heatmap(heatmap, plane_idx, center, sigma):
        center_x, center_y = center
        _, height, width = heatmap.shape[:3]

        th = 4.6052
        delta = math.sqrt(th * 2)

        x0 = int(max(0, center_x - delta * sigma))
        y0 = int(max(0, center_y - delta * sigma))

        x1 = int(min(width, center_x + delta * sigma))
        y1 = int(min(height, center_y + delta * sigma))

        for y in range(y0, y1):
            for x in range(x0, x1):
                d = (x - center_x) ** 2 + (y - center_y) ** 2
                exp = d / 2.0 / sigma / sigma
                if exp > th:
                    continue
                heatmap[plane_idx][y][x] = max(heatmap[plane_idx][y][x], math.exp(-exp))
                heatmap[plane_idx][y][x] = min(heatmap[plane_idx][y][x], 1.0)

    def get_vectormap(self, target_size):
        vectormap = np.zeros((CocoMetadata.__coco_parts*2, self.height, self.width))
        countmap = np.zeros((CocoMetadata.__coco_parts, self.height, self.width))
        for joints in self.joint_list:
            for plane_idx, (j_idx1, j_idx2) in enumerate(CocoMetadata.__coco_vecs):
                j_idx1 -= 1
                j_idx2 -= 1

                center_from = joints[j_idx1]
                center_to = joints[j_idx2]

                if center_from[0] < -100 or center_from[1] < -100 or center_to[0] < -100 or center_to[1] < -100:
                    continue

                CocoMetadata.put_vectormap(vectormap, countmap, plane_idx, center_from, center_to)

        vectormap = vectormap.transpose((1, 2, 0))
        nonzeros = np.nonzero(countmap)
        for p, y, x in zip(nonzeros[0], nonzeros[1], nonzeros[2]):
            if countmap[p][y][x] <= 0:
                continue
            vectormap[y][x][p*2+0] /= countmap[p][y][x]
            vectormap[y][x][p*2+1] /= countmap[p][y][x]

        if target_size:
            vectormap = cv2.resize(vectormap, target_size, interpolation=cv2.INTER_AREA)

        return vectormap

    @staticmethod
    def put_vectormap(vectormap, countmap, plane_idx, center_from, center_to, threshold=8):
        _, height, width = vectormap.shape[:3]

        vec_x = center_to[0] - center_from[0]
        vec_y = center_to[1] - center_from[1]

        min_x = max(0, int(min(center_from[0], center_to[0]) - threshold))
        min_y = max(0, int(min(center_from[1], center_to[1]) - threshold))

        max_x = min(width, int(max(center_from[0], center_to[0]) + threshold))
        max_y = min(height, int(max(center_from[1], center_to[1]) + threshold))

        norm = math.sqrt(vec_x ** 2 + vec_y ** 2)
        if norm == 0:
            return

        vec_x /= norm
        vec_y /= norm

        for y in range(min_y, max_y):
            for x in range(min_x, max_x):
                bec_x = x - center_from[0]
                bec_y = y - center_from[1]
                dist = abs(bec_x * vec_y - bec_y * vec_x)

                if dist > threshold:
                    continue

                countmap[plane_idx][y][x] += 1

                vectormap[plane_idx*2+0][y][x] = vec_x
                vectormap[plane_idx*2+1][y][x] = vec_y


class CocoPoseLMDB(RNGDataFlow):
    __valid_i = 2745
    __max_key = 121745

    @staticmethod
    def display_image(inp, heatmap, vectmap, as_numpy=False):
        if as_numpy:
            mpl.use('Agg')
        import matplotlib.pyplot as plt

        fig = plt.figure()
        a = fig.add_subplot(2, 2, 1)
        a.set_title('Image')
        plt.imshow(CocoPoseLMDB.get_bgimg(inp))

        a = fig.add_subplot(2, 2, 2)
        a.set_title('Heatmap')
        plt.imshow(CocoPoseLMDB.get_bgimg(inp, target_size=(heatmap.shape[1], heatmap.shape[0])), alpha=0.5)
        tmp = np.amax(heatmap, axis=2)
        plt.imshow(tmp, cmap=plt.cm.gray, alpha=0.5)
        plt.colorbar()

        tmp2 = vectmap.transpose((2, 0, 1))
        tmp2_odd = np.amax(np.absolute(tmp2[::2, :, :]), axis=0)
        tmp2_even = np.amax(np.absolute(tmp2[1::2, :, :]), axis=0)

        a = fig.add_subplot(2, 2, 3)
        a.set_title('Vectormap-x')
        plt.imshow(CocoPoseLMDB.get_bgimg(inp, target_size=(vectmap.shape[1], vectmap.shape[0])), alpha=0.5)
        plt.imshow(tmp2_odd, cmap=plt.cm.gray, alpha=0.5)
        plt.colorbar()

        a = fig.add_subplot(2, 2, 4)
        a.set_title('Vectormap-y')
        plt.imshow(CocoPoseLMDB.get_bgimg(inp, target_size=(vectmap.shape[1], vectmap.shape[0])), alpha=0.5)
        plt.imshow(tmp2_even, cmap=plt.cm.gray, alpha=0.5)
        plt.colorbar()

        if not as_numpy:
            plt.show()
        else:
            fig.canvas.draw()
            data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
            data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            fig.clear()
            plt.close()
            return data

    @staticmethod
    def get_bgimg(inp, target_size=None):
        if target_size:
            inp = cv2.resize(inp, target_size, interpolation=cv2.INTER_AREA)
        inp = cv2.cvtColor(((inp + 1.0) * (255.0 / 2.0)).astype(np.uint8), cv2.COLOR_BGR2RGB)
        return inp

    def __init__(self, path, is_train=True, decode_img=True, only_idx=-1):
        self.is_train = is_train
        self.decode_img = decode_img
        self.only_idx = only_idx
        self.env = lmdb.open(path, map_size=int(1e12), readonly=True)
        self.txn = self.env.begin(buffers=True)
        pass

    def size(self):
        if self.is_train:
            return CocoPoseLMDB.__max_key - CocoPoseLMDB.__valid_i
        else:
            return CocoPoseLMDB.__valid_i

    def get_data(self):
        idxs = np.arange(self.size())
        if self.is_train:
            idxs += CocoPoseLMDB.__valid_i
            self.rng.shuffle(idxs)
        else:
            pass

        for idx in idxs:
            datum = Datum()
            if self.only_idx < 0:
                s = self.txn.get(('%07d' % idx).encode('utf-8'))
            else:
                s = self.txn.get(('%07d' % self.only_idx).encode('utf-8'))
            datum.ParseFromString(s)
            if isinstance(datum.data, bytes):
                data = np.fromstring(datum.data, dtype=np.uint8).reshape(datum.channels, datum.height, datum.width)
            else:
                data = np.fromstring(datum.data.tobytes(), dtype=np.uint8).reshape(datum.channels, datum.height,
                                                                                   datum.width)
            if self.decode_img:
                img = data[:3].transpose((1, 2, 0))
            else:
                img = None

            meta = CocoMetadata(idx, img, data[3], sigma=8.0)

            yield [meta]


def get_dataflow(path, is_train):
    ds = CocoPoseLMDB(path, is_train)       # read data from lmdb
    if is_train:
        ds = MapDataComponent(ds, pose_random_scale)
        ds = MapDataComponent(ds, pose_rotation)
        ds = MapDataComponent(ds, pose_flip)
        ds = MapDataComponent(ds, pose_resize_shortestedge_random)
        ds = MapDataComponent(ds, pose_crop_random)
        ds = MapData(ds, pose_to_img)
        augs = [
            imgaug.RandomApplyAug(imgaug.RandomChooseAug([
                imgaug.BrightnessScale((0.6, 1.4), clip=False),
                imgaug.Contrast((0.7, 1.4), clip=False),
                imgaug.GaussianBlur(max_size=3)
            ]), 0.7),
        ]
        ds = AugmentImageComponent(ds, augs)
    else:
        ds = MapDataComponent(ds, pose_resize_shortestedge_fixed)
        ds = MapDataComponent(ds, pose_crop_center)
        ds = MapData(ds, pose_to_img)

    ds = PrefetchData(ds, 1000, multiprocessing.cpu_count())

    return ds


def get_dataflow_batch(path, is_train, batchsize):
    ds = get_dataflow(path, is_train)
    ds = BatchData(ds, batchsize)
    ds = PrefetchData(ds, 10, 2)

    return ds


class DataFlowToQueue(threading.Thread):
    def __init__(self, ds, placeholders, queue_size=5):
        super().__init__()
        self.daemon = True

        self.ds = ds
        self.placeholders = placeholders
        self.queue = tf.FIFOQueue(queue_size, [ph.dtype for ph in placeholders], shapes=[ph.get_shape() for ph in placeholders])
        self.op = self.queue.enqueue(placeholders)
        self.close_op = self.queue.close(cancel_pending_enqueues=True)

        self._coord = None
        self._sess = None

        self.last_dp = None

    @contextmanager
    def default_sess(self):
        if self._sess:
            with self._sess.as_default():
                yield
        else:
            logging.warning("DataFlowToQueue {} wasn't under a default session!".format(self.name))
            yield

    def size(self):
        return self.queue.size()

    def start(self):
        self._sess = tf.get_default_session()
        super().start()

    def set_coordinator(self, coord):
        self._coord = coord

    def run(self):
        with self.default_sess():
            try:
                while not self._coord.should_stop():
                    try:
                        self.ds.reset_state()
                        while True:
                            for dp in self.ds.get_data():
                                feed = dict(zip(self.placeholders, dp))
                                self.op.run(feed_dict=feed)
                                self.last_dp = dp
                    except (tf.errors.CancelledError, tf.errors.OutOfRangeError, DataFlowTerminated):
                        logging.error('err type1, placeholders={}'.format(self.placeholders))
                        sys.exit(-1)
                    except Exception as e:
                        logging.error('err type2, err={}, placeholders={}'.format(str(e), self.placeholders))
                        if isinstance(e, RuntimeError) and 'closed Session' in str(e):
                            pass
                        else:
                            logging.exception("Exception in {}:{}".format(self.name, str(e)))
                        sys.exit(-1)
            except Exception as e:
                logging.exception("Exception in {}:{}".format(self.name, str(e)))
            finally:
                try:
                    self.close_op.run()
                except Exception:
                    pass
                logging.info("{} Exited.".format(self.name))

    def dequeue(self):
        return self.queue.dequeue()


if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = ''

    from pose_augment import set_network_input_wh
    set_network_input_wh(368, 368)

    # df = get_dataflow('/data/public/rw/coco-pose-estimation-lmdb/', False)
    df = get_dataflow('/data/public/rw/coco-pose-estimation-lmdb/', True)

    # input_node = tf.placeholder(tf.float32, shape=(None, 368, 368, 3), name='image')
    with tf.Session() as sess:
        # net = CmuNetwork({'image': input_node}, trainable=False)
        # net.load('./models/numpy/openpose_coco.npy', sess)

        df.reset_state()
        t1 = time.time()
        for idx, dp in enumerate(df.get_data()):
            if idx == 0:
                for d in dp:
                    logging.info('%d dp shape={}'.format(d.shape))
            if idx % 100 == 0:
                print(time.time() - t1)
                t1 = time.time()
                CocoPoseLMDB.display_image(dp[0], dp[1], dp[2])
                print(dp[1].shape, dp[2].shape)

                # pafMat, heatMat = sess.run(net.loss_last(), feed_dict={'image:0': [dp[0] / 128.0]})
                # print(heatMat.shape, pafMat.shape)
                # CocoPoseLMDB.display_image(dp[0], heatMat[0], pafMat[0])
            pass

    logging.info('done')
