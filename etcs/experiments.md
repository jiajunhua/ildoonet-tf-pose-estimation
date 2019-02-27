
## Models

- mobilenet_thin
- mobilenet_v2_large
- mobilenet_v2_normal
- mobilenet_v2_fast
- mobilenet_v2_faster

## Performance on COCO Datasets

| Set         | Model              | Scale      | Resolution | AP         | AP 50      | AP 75      | AP medium  | AP large   | AR         | AR 50      | AR 75      | AR medium  | AR large   |
|-------------|--------------------|------------|------------|------------|------------|------------|------------|------------|------------|------------|------------|------------|------------|
| 2014 Val    | Original Paper     | 3          | Image      |      0.584 |      0.815 |      0.626 |      0.544 |      0.651 |            |            |            |            |            |
| | | | | | | | | | | | | |
| 2014 Val    | CMU(openpose)      | 1          | Image      |     0.5067 |     0.7660 |     0.5377 |     0.4927 |     0.5309 |     0.5614 |     0.7900 |     0.5903 |     0.5089 |     0.6347 |
| 2014 Val    | VGG(openpose, our) | 1          | Image      |     0.5067 |     0.7660 |     0.5377 |     0.4927 |     0.5309 |     0.5614 |     0.7900 |     0.5903 |     0.5089 |     0.6347 |
| | | | | | | | | | | | | |
| 2014 Val    | Mobilenet thin     | 1          | Image      |     0.2806 |     0.5577 |     0.2474 |     0.2802 |     0.2843 |     0.3214 |     0.5840 |     0.2997 |     0.2946 |     0.3587 |
| 2014 Val    | Mobilenetv2 Large  | 1          | Image      |     0.3130 |     0.5846 |     0.2940 |     0.2622 |     0.3850 |     0.3680 |     0.6101 |     0.3637 |     0.2765 |     0.4912 |
| 2014 Val    | Mobilenetv2 Normal | 1          | Image      |     0.2838 |     0.5456 |     0.2591 |     0.2421 |     0.3490 |     0.3355 |     0.5762 |     0.3259 |     0.2548 |     0.4441 |

I also ran keras & caffe models to verify single-scale version's performance, they matched this result.

## Computation Budget & Latency

| Model               | mAP@COCO2014 | GFLOPs | Latency(432x368)<br/>(Macbook 15' 2.9GHz i9, tf 1.12) |
|---------------------|--------------|--------|-------------------------------------------------------|
| CMU, VGG(OpenPose)  |              |        | 0.8589s |
| Mobilenet thin      | 0.2806       |        | 0.1701s |
| Mobilenetv2 Large   | 0.3130       |        | 0.2066s |
| Mobilenetv2 Normal  | 0.2838       |        | 0.1813s |