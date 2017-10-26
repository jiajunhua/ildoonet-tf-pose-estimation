import unittest

import logging
import numpy as np
import cv2
import time

import common
from pose_dataset import CocoPoseLMDB

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')


class TestStringMethods(unittest.TestCase):
    def _show(self, path, inpmat, heatmat, pafmat, humans):
        image = cv2.imread(path)

        # CocoPoseLMDB.display_image(inpmat, heatmat, pafmat)

        image_h, image_w = image.shape[:2]
        heat_h, heat_w = heatmat.shape[:2]
        for _, human in humans.items():
            for part in human:
                if part['partIdx'] not in common.CocoPairsRender:
                    continue
                center1 = (int((part['c1'][0] + 0.5) * image_w / heat_w), int((part['c1'][1] + 0.5) * image_h / heat_h))
                center2 = (int((part['c2'][0] + 0.5) * image_w / heat_w), int((part['c2'][1] + 0.5) * image_h / heat_h))
                cv2.circle(image, center1, 2, (255, 0, 0), thickness=3, lineType=8, shift=0)
                cv2.circle(image, center2, 2, (255, 0, 0), thickness=3, lineType=8, shift=0)
                cv2.putText(image, str(part['partIdx'][1]), center2, cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 0, 0), 1)
                image = cv2.line(image, center1, center2, (255, 0, 0), 1)
        cv2.imshow('result', image)
        cv2.waitKey(0)

    def test_mobilenet(self):
        inpmat = np.load('./tests/person3.pickle')
        heatmat = np.load('./tests/mobilenet_person3_heatmat.pickle')
        pafmat = np.load('./tests/mobilenet_person3_pafmat.pickle')

        t = time.time()
        humans = common.estimate_pose(heatmat, pafmat)
        elapsed = time.time() - t
        logging.info('[test_mobilenet] elapsed=%f' % elapsed)

        self._show('./images/p3.jpg', inpmat, heatmat, pafmat, humans)

    def test_cmu(self):
        inpmat = np.load('./tests/person3.pickle')
        heatmat = np.load('./tests/cmu_person3_heatmat.pickle')
        pafmat = np.load('./tests/cmu_person3_pafmat.pickle')

        t = time.time()
        humans = common.estimate_pose(heatmat, pafmat)
        elapsed = time.time() - t
        logging.info('[test_mobilenet] elapsed=%f' % elapsed)

        # self._show('./images/p3.jpg', inpmat, heatmat, pafmat, humans)

if __name__ == '__main__':
    unittest.main()
