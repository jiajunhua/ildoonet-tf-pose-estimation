import math
import struct
import cv2

import lmdb
import logging

import multiprocessing
import numpy as np
import matplotlib.pyplot as plt

from tensorpack import imgaug
from tensorpack.dataflow.image import MapDataComponent, AugmentImageComponent
from tensorpack.dataflow.common import BatchData, MapData
from tensorpack.dataflow.prefetch import PrefetchData
from tensorpack.dataflow.base import RNGDataFlow, DataFlowTerminated

from datum_pb2 import Datum
from pose_augment import pose_flip, pose_rotation, pose_to_img, pose_crop_random, \
    pose_resize_shortestedge_random, pose_resize_shortestedge_fixed, pose_crop_center

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

    def __init__(self, img, meta, sigma):
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
                    new_joint.append((-1, -1))
                else:
                    new_joint.append(((j1[0] + j2[0]) / 2, (j1[1] + j2[1]) / 2))

            new_joint.append((-1, -1))
            self.joint_list.append(new_joint)

        logging.debug('joint size=%d' % len(self.joint_list))

    def get_heatmap(self, target_size=None):
        heatmap = np.zeros((CocoMetadata.__coco_parts + 1, self.height, self.width))

        for joints in self.joint_list:
            for idx, point in enumerate(joints):
                if point[0] < 0 or point[1] < 0:
                    continue
                CocoMetadata.put_heatmap(heatmap, idx, point, self.sigma)

        heatmap = heatmap.transpose((1, 2, 0))

        # background
        height, width = heatmap.shape[:2]
        for y in range(height):
            for x in range(width):
                maximum = max(heatmap[y][x])
                heatmap[y][x][-1] = max(1.0 - maximum, 0.0)

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
                heatmap[plane_idx][y][x] += math.exp(-exp)
                heatmap[plane_idx][y][x] = min(heatmap[plane_idx][y][x], 1.0)

    def get_vectormap(self, target_size=None):
        vectormap = np.zeros((CocoMetadata.__coco_parts*2, self.height, self.width))
        countmap = np.zeros((CocoMetadata.__coco_parts, self.height, self.width))
        for joints in self.joint_list:
            for plane_idx, (j_idx1, j_idx2) in enumerate(CocoMetadata.__coco_vecs):
                j_idx1 -= 1
                j_idx2 -= 1

                center_from = joints[j_idx1]
                center_to = joints[j_idx2]

                if center_from[0] < 0 or center_from[1] < 0 or center_to[0] < 0 or center_to[1] < 0:
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
    def put_vectormap(vectormap, countmap, plane_idx, center_from, center_to, threshold=4):
        _, height, width = vectormap.shape[:3]

        vec_x = center_to[0] - center_from[0]
        vec_y = center_to[1] - center_from[1]

        min_x = max(0, int(min(center_from[0], center_to[0]) - threshold))
        min_y = max(0, int(min(center_from[1], center_to[1]) - threshold))

        max_x = min(width, int(max(center_from[0], center_to[0]) + threshold))
        max_y = min(height, int(max(center_from[1], center_to[1]) + threshold))

        norm = math.sqrt(vec_x ** 2 + vec_y ** 2)
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
    def display_image(inp, heatmap, vectmap):
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
        tmp2_odd = np.amax(tmp2[::2, :, :], axis=0)
        tmp2_even = np.amax(tmp2[1::2, :, :], axis=0)

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

        plt.show()

    @staticmethod
    def get_bgimg(inp, target_size=None):
        if target_size:
            inp = cv2.resize(inp, target_size, interpolation = cv2.INTER_AREA)
        inp = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)
        return inp

    def __init__(self, path, is_train=True):
        self.is_train = is_train
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
            s = self.txn.get(('%07d' % idx).encode('utf-8'))
            datum.ParseFromString(s)
            data = np.fromstring(datum.data.tobytes(), dtype=np.uint8).reshape(datum.channels, datum.height, datum.width)
            img = data[:3].transpose((1, 2, 0))

            meta = CocoMetadata(img, data[3], 4.0)

            yield [meta]


def get_dataflow(is_train):
    ds = CocoPoseLMDB('/data/public/rw/coco-pose-estimation-lmdb/', is_train)
    if is_train:
        ds = MapDataComponent(ds, pose_rotation)
        ds = MapDataComponent(ds, pose_flip)
        ds = MapDataComponent(ds, pose_resize_shortestedge_random)
        ds = MapDataComponent(ds, pose_crop_random)
        ds = MapData(ds, pose_to_img)
        augs = [
            imgaug.RandomApplyAug(imgaug.RandomChooseAug([
                imgaug.SaltPepperNoise(white_prob=0.01, black_prob=0.01),
                imgaug.RandomOrderAug([
                    imgaug.BrightnessScale((0.8, 1.2), clip=False),
                    imgaug.Contrast((0.8, 1.2), clip=False),
                    # imgaug.Saturation(0.4, rgb=True),
                ]),
            ]), 0.7),
        ]
        ds = AugmentImageComponent(ds, augs)
    else:
        ds = MapDataComponent(ds, pose_resize_shortestedge_fixed)
        ds = MapDataComponent(ds, pose_crop_center)
        ds = MapData(ds, pose_to_img)

    return ds


def get_dataflow_batch(is_train, batchsize):
    ds = get_dataflow(is_train)
    ds = PrefetchData(ds, 1000, multiprocessing.cpu_count())
    ds = BatchData(ds, batchsize)
    ds = PrefetchData(ds, 10, 4)

    return ds


if __name__ == '__main__':
    df = get_dataflow(False)

    df.reset_state()
    for dp in df.get_data():
        CocoPoseLMDB.display_image(dp[0], dp[1], dp[2])
        pass

    logging.info('done')
