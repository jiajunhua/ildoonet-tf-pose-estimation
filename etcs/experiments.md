

## COCO Datasets

| Set         | Model          | Scale      | Resolution | AP         | AP 50      | AP 75      | AP medium  | AP large   | AR         | AR 50      | AR 75      | AR medium  | AR large   |
|-------------|----------------|------------|------------|------------|------------|------------|------------|------------|------------|------------|------------|------------|------------|
| 2014 Val    | Original Paper | 3          | Image      |      0.584 |      0.815 |      0.626 |      0.544 |      0.651 |            |            |            |            |            |
| | | | | | | | | | | | | |
| 2014 Val    | CMU            | 1          | Image      |    0.5067 |     0.7660 |     0.5377 |     0.4927 |     0.5309 |     0.5614 |     0.7900 |     0.5903 |     0.5089 |     0.6347 |
| 2014 Val    | Mobilenet thin | 1          | Image      |    0.2806 |     0.5577 |     0.2474 |     0.2802 |     0.2843 |     0.3214 |     0.5840 |     0.2997 |     0.2946 |     0.3587 |

I also ran keras & caffe models to verify single-scale version's performance, they matched this result.
