# tf-openpose

Openpose from CMU implemented using Tensorflow. It also provides several variants that have made some changes to the network structure for **real-time processing on the CPU.**

Original Repo(Caffe) : https://github.com/CMU-Perceptual-Computing-Lab/openpose

**Features**

- [x] CMU's original network architecture and weights.

  - [x] Transfer Original Weights to Tensorflow

  - [x] Training Code with multi-gpus
  
  - [ ] Evaluate with test dataset

- [ ] Inference

  - [ ] Post processing from network output.

  - [ ] Multi-Scale Inference

- [ ] Faster network variants using mobilenet, lcnn architecture.

  - [x] Depthwise Separable Convolution Version
  
  - [ ] Mobilenet Version
  
  - [ ] LCNN Version

- [ ] Demos

  - [ ] Realtime Webcam Demo
  
  - [ ] Image/Video File Demo

- [ ] ROS Support. 

## Install

### Dependencies

You need dependencies below.

- python3

- tensorflow 1.3

- opencv3

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

| Dataset | Model              | System               | Description                                                                                  | Inference Time |
|---------|--------------------|----------------------|----------------------------------------------------------------------------------------------|---------------:|
| Coco    | cmu                | 3.1 GHz i5 Dual Core | CMU's original version. Same network, same weights.                                          | 10s @ 368x368 <br/>3.65s @ 320x240 |
| Coco    | dsconv             | 3.1 GHz i5 Dual Core | Same architecture as the cmu version except for<br/>the **depthwise separable convolution** of mobilenet. | 1.1s @ 368x368<br/> 0.44s @ 320x240 |
| Coco    | mobilenet          | 3.1 GHz i5 Dual Core | Feature extraction layers is replaced from VGG to **Mobilenet** from Google                  | 0.2s @ 368x368 |

* Test being processed. This will be updated soon.

## Training

### Dataset

### Augmentation

CMU Perceptual Computing Lab has modified Caffe to provide data augmentation. See : https://github.com/CMU-Perceptual-Computing-Lab/caffe_train

I implemented the augmentation codes as the way of the original version, See [pose_dataset.py](pose_dataset.py) and [pose_augment.py](pose_augment.py). This includes scaling, rotation, flip, cropping.

This process can be a bottleneck for training, so if you have enough computing resources, please see [Run for Faster Training]() Section

### Run

```
$ python3 train.py --model=cmu --datapath={datapath} --batchsize=64 --lr=0.001 --modelpath={path-to-save}

2017-09-27 15:58:50,307 INFO Restore pretrained weights...
```

### Run for Faster Training

If you have enough computing resources in multiple nodes, you can launch multiple workers on nodes to help data preparation.
 
```
worker-node1$ python3 pose_dataworker.py --master=tcp://host:port
worker-node2$ python3 pose_dataworker.py --master=tcp://host:port
worker-node3$ python3 pose_dataworker.py --master=tcp://host:port
...
```

After above preparation, you can launch training script with special arguments.

```
$ python3 train.py --remote-data=tcp://0.0.0.0:port

2017-09-27 15:58:50,307 INFO Restore pretrained weights...
```

Also, You can quickly train with multiple gpus. This automatically splits batch into multiple gpus for forward/backward computations.

```
$ python3 train.py --remote-data=tcp://0.0.0.0:port --gpus=8

2017-09-27 15:58:50,307 INFO Restore pretrained weights...
```

I trained models within a day with 8 gpus and multiple preprocessing nodes with 48 core cpus.

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