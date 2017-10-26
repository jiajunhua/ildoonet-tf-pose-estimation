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