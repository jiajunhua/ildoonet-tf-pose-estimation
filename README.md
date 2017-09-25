# tf-openpose

Openpose from CMU implemented using Tensorflow. It also provides several variants that have made some changes to the network structure for **real-time processing on the CPU.**

Original Repo(Caffe) : https://github.com/CMU-Perceptual-Computing-Lab/openpose

**Features**

- [x] CMU's original network architecture and weights.

- [ ] Post processing from network output.

- [ ] Faster network variants using mobilenet, lcnn architecture.

- [ ] ROS Support. 

## Install

### Dependencies

You need dependencies below.

- python3

- tensorflow 1.3

- opencv 3

- protobuf

- python3-tk

### Install

```bash
$ git clone https://www.github.com/ildoonet/tf-openpose
$ cd tf-openpose
$ git lfs pull
$ pip3 install -r requirements.txt
```

## Models

### Inference Time

| Dataset | Model                             | Description                                                                     | Inference Time<br/>1 core cpu |
|---------|--------------------------|------------------------------------------------------------------------------------------|---------------:|
| Coco    | cmu                      | CMU's original version. Same network, same weights.                                      | 368x368 @ 10s<br/>320x240 @ 3.65s |
| Coco    | dsconv                   | Same architecture as the cmu version except for<br/>the **depthwise separable convolution** of mobilenet. | 368x368 @ 1.1s<br/>320x240 @ 0.44s  |
| Coco    | mobilenet                | Feature extraction layers is replaced from VGG to Mobilenet from Google                  | | |
| Coco    | lcnn      | | | |


## Training

### Dataset

### Augmentation

CMU Perceptual Computing Lab has modified Caffe to provide data augmentation. See : https://github.com/CMU-Perceptual-Computing-Lab/caffe_train

I implemented the augmentation codes as the way of the original version, See [pose_dataset.py](pose_dataset.py) and [pose_augment.py](pose_augment.py). This includes scaling, rotation, flip, cropping.

### Run


## References

### OpenPose

[1] https://github.com/CMU-Perceptual-Computing-Lab/openpose

[2] Training Codes : https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation

[3] Custom Caffe by Openpose : https://github.com/CMU-Perceptual-Computing-Lab/caffe_train

### Mobilenet

[1] Original Paper : https://arxiv.org/abs/1704.04861

[2] Pretrained model : https://github.com/tensorflow/models/blob/master/slim/nets/mobilenet_v1.md

### Libraries

[1] Tensorpack : https://github.com/ppwwyyxx/tensorpack