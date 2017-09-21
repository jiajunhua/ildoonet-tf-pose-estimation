# tf-openpose

Openpose from CMU implemented using Tensorflow. It also provides several variants that have made some changes to the network structure for **real-time processing on the CPU.**

Original Repo(Caffe) : https://github.com/CMU-Perceptual-Computing-Lab/openpose

**Features**

[x] CMU's original network architecture and weights.

[] Post processing from network output.

[] Faster network variants using mobilenet, lcnn architecture.

[] ROS Support. 

## Install

You need dependencies below.

- python3

- tensorflow 1.3

- opencv 3

- protobuf

## Models

### Inference Time

| Dataset | Model                             | Description                                                                              | Inference Time<br/>Single core cpu |
|---------|-----------------------------------|------------------------------------------------------------------------------------------|---------------:|
| Coco    | openopse-cmu                      | CMU's original version. Same network, same weights.                                      | 3.65s / image  |
| Coco    | openpose-depthwise-separable-conv | Same as the cmu version except for the **depthwise separable convolution** of mobilenet. | 0.44s / image  |
| Coco    | openpose-mobilenet                | | | |
| Coco    | openpose-lcnn      | | | |


## Training

CMU Perceptual Computing Lab has modified Caffe to provide data augmentation.

This includes

- scale : 0.7 ~ 1.3

- rotation : -40 ~ 40 degrees

- flip

- cropping

See : https://github.com/CMU-Perceptual-Computing-Lab/caffe_train



## References

### OpenPose

[1] https://github.com/CMU-Perceptual-Computing-Lab/openpose

[2] Training Codes : https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation

[3] Custom Caffe by Openpose : https://github.com/CMU-Perceptual-Computing-Lab/caffe_train

### Mobilenet

[2] Pretrained model : https://github.com/tensorflow/models/blob/master/slim/nets/mobilenet_v1.md

### Libraries

[1] Tensorpack : https://github.com/ppwwyyxx/tensorpack