# tf-openpose

'Openpose' for human pose estimation have been implemented using Tensorflow. It also provides several variants that have made some changes to the network structure for **real-time processing on the CPU or low-power embedded devices.**


**You can even run this on your macbook with descent FPS!**

Original Repo(Caffe) : https://github.com/CMU-Perceptual-Computing-Lab/openpose

| CMU's Original Model</br> on Macbook Pro 15" | Mobilenet Variant </br>on Macbook Pro 15" | Mobilenet Variant</br>on Jetson TK2 |
|:---------|:--------------------|:----------------|
| ![cmu-model](/etcs/openpose_macbook_cmu.gif)     | ![mb-model-macbook](/etcs/openpose_macbook_mobilenet3.gif) | ![mb-model-tx2](/etcs/openpose_tx2_mobilenet3.gif) |
| **~0.6 FPS** | **~4.2 FPS** @ 368x368 | **~10 FPS** @ 368x368 |
| 2.8GHz Quad-core i7 | 2.8GHz Quad-core i7 | Jetson TX2 Embedded Board | 

Implemented features are listed here : [features](./etcs/feature.md)

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
$ pip3 install -r requirements.txt
```

## Models

- cmu 
  - the model based VGG pretrained network which described in the original paper.
  - I converted Weights in Caffe format to use in tensorflow.
  - [weight download](https://www.dropbox.com/s/xh5s7sb7remu8tx/openpose_coco.npy?dl=0)
  
- dsconv
  - Same architecture as the cmu version except for<br/>the **depthwise separable convolution** of mobilenet.
  - I trained it using 'transfer learning', but it provides not-enough speed and accuracy.
  
- mobilenet
  - Based on the mobilenet paper, 12 convolutional layers are used as feature-extraction layers.
  - To improve on small person, **minor modification** on the architecture have been made.
  - Three models were learned according to network size parameters.
    - mobilenet
      - 368x368 : [weight download](https://www.dropbox.com/s/09xivpuboecge56/mobilenet_0.75_0.50_model-388003.zip?dl=0)
    - mobilenet_fast
    - mobilenet_accurate
  - I published models which is not the best ones, but you can test them before you trained a model from the scratch.

### Inference Time

#### Macbook Pro - 3.1GHz i5 Dual Core

| Dataset | Model              | Inference Time  |
|---------|--------------------|----------------:|
| Coco    | cmu                | 10.0s @ 368x368 |
| Coco    | dsconv             | 1.10s @ 368x368 |
| Coco    | mobilenet_accurate | 0.40s @ 368x368 |
| Coco    | mobilenet          | 0.24s @ 368x368 |
| Coco    | mobilenet_fast     | 0.16s @ 368x368 |

#### Jetson TX2

On embedded GPU Board from Nvidia, Test results are as below.

| Dataset | Model              | Inference Time  |
|---------|--------------------|----------------:|
| Coco    | cmu                | OOM   @ 368x368<br/> 5.5s  @ 320x240|
| Coco    | mobilenet_accurate | 0.18s @ 368x368 |
| Coco    | mobilenet          | 0.10s @ 368x368 |
| Coco    | mobilenet_fast     | 0.07s @ 368x368 |

CMU's original model can not be executed due to 'out of memory' on '368x368' size.

## Demo

### Test Inference

You can test the inference feature with a single image.

```
$ python3 inference.py --model=mobilenet --imgpath=...
```

Then you will see the screen as below with pafmap, heatmap, result and etc.

![inferent_result](./etcs/inference_result2.png)

### Realtime Webcam

```
$ python3 realtime_webcam.py --camera=0 --model=mobilenet --zoom=1.0
```

Then you will see the realtime webcam screen with estimated poses as below. This [Realtime Result](./etcs/openpose_macbook13_mobilenet2.gif) was recored on macbook pro 13" with 3.1Ghz Dual-Core CPU.

## Training

See : [etcs/training.md](./etcs/training.md)

## References

### OpenPose

[1] https://github.com/CMU-Perceptual-Computing-Lab/openpose

[2] Training Codes : https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation

[3] Custom Caffe by Openpose : https://github.com/CMU-Perceptual-Computing-Lab/caffe_train

[4] Keras Openpose : https://github.com/michalfaber/keras_Realtime_Multi-Person_Pose_Estimation

### Mobilenet

[1] Original Paper : https://arxiv.org/abs/1704.04861

[2] Pretrained model : https://github.com/tensorflow/models/blob/master/slim/nets/mobilenet_v1.md

### Libraries

[1] Tensorpack : https://github.com/ppwwyyxx/tensorpack

### Tensorflow Tips

[1] Freeze graph : https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/tools/freeze_graph.py

[2] Optimize graph : https://codelabs.developers.google.com/codelabs/tensorflow-for-poets-2
