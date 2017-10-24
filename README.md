# tf-openpose

'Openpose' for human pose estimation have been implemented using Tensorflow. It also provides several variants that have made some changes to the network structure for **real-time processing on the CPU or low-power embedded devices.**

Original Repo(Caffe) : https://github.com/CMU-Perceptual-Computing-Lab/openpose

| CMU's Original Model</br> on Macbook Pro 15" | Mobilenet Variant </br>on Macbook Pro 15" | Mobilenet Variant</br>on Jetson TK2 |
|:---------|:--------------------|:----------------|
| ![cmu-model](/etcs/openpose_macbook_cmu.gif)     | ![mb-model-macbook](/etcs/openpose_macbook_mobilenet.gif)  | ![mb-model-tx2](/etcs/openpose_tx2_mobilenet.gif) |
| **~0.6 FPS** | **~4.2 FPS** | **~10 FPS** |
| 2.8GHz Quad-core i7 | 2.8GHz Quad-core i7 | Jetson TX2 Embedded Board | 

**Features**

- [x] CMU's original network architecture and weights.

  - [x] Transfer Original Weights to Tensorflow

  - [x] Training Code with multi-gpus
  
  - [ ] Evaluate with test dataset

- [ ] Inference

  - [x] Post processing from network output.

  - [ ] Faster post-processing

  - [ ] Multi-Scale Inference

- [x] Faster network variants using custom mobilenet architecture.

  - [x] Depthwise Separable Convolution Version
  
  - [x] Mobilenet Version
  
- [ ] Demos

  - [x] Realtime Webcam Demo
  
  - [x] Image File Demo
  
  - [ ] Video File Demo

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
    - mobilenet : [weight download](https://www.dropbox.com/s/izh6a3xsn7pdxd0/mobilenet_0.75_0.50_model-146000.zip?dl=0)
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
| Coco    | cmu                | OOM   @ 368x368 |
| Coco    | mobilenet_accurate | 0.18s @ 368x368 |
| Coco    | mobilenet          | 0.10s @ 368x368 |
| Coco    | mobilenet_fast     | 0.07s @ 368x368 |

CMU's original model can not be executed due to 'out of memory' on '368x368' size.

## Training

### Dataset

You should download the dataset in LMDB format provided by CMU. See : https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation/blob/master/training/get_lmdb.sh

```
$ wget -nc --directory-prefix=lmdb_trainVal/ 		http://posefs1.perception.cs.cmu.edu/Users/ZheCao/lmdb_trainVal/data.mdb
$ wget -nc --directory-prefix=lmdb_trainVal/ 		http://posefs1.perception.cs.cmu.edu/Users/ZheCao/lmdb_trainVal/lock.mdb
```

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

After above preparation, you can launch training script with 'remote-data' arguments.

```
$ python3 train.py --remote-data=tcp://0.0.0.0:port

2017-09-27 15:58:50,307 INFO Restore pretrained weights...
```

Also, You can quickly train with multiple gpus. This automatically splits batch into multiple gpus for forward/backward computations.

```
$ python3 train.py --remote-data=tcp://0.0.0.0:port --gpus=8

2017-09-27 15:58:50,307 INFO Restore pretrained weights...
```

I trained models within a day with 8 gpus and multiple pre-processing nodes with 48 core cpus.

### Model Optimization for Inference

After trained a model, I optimized models by folding batch normalization to convolutional layers and removing redundant operations.  

Firstly, the model should be frozen.

```bash
$ python3 -m tensorflow.python.tools.freeze_graph \
  --input_graph=... \
  --output_graph=... \
  --input_checkpoint=... \
  --output_node_names="Openpose/concat_stage7"
```

And the optimization can be performed on the frozen model via graph transform provided by tensorflow. 

```bash
$ bazel build tensorflow/tools/graph_transforms:transform_graph
$ bazel-bin/tensorflow/tools/graph_transforms/transform_graph \
    --in_graph=... \
    --out_graph=... \
    --inputs='image:0' \
    --outputs='Openpose/concat_stage7:0' \
    --transforms='
    strip_unused_nodes(type=float, shape="1,368,368,3")
    remove_nodes(op=Identity, op=CheckNumerics)
    fold_constants(ignoreError=False)
    fold_old_batch_norms
    fold_batch_norms'
```

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
