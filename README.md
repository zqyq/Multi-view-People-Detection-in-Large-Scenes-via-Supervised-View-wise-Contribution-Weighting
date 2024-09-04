#  Multi-view People Detection in Large Scenes via Supervised View-wise Contribution Weighting, AAAI 2024

![Pipeline](https://github.com/zqyq/Multi-view-People-Detection-in-Large-Scenes-via-Supervised-View-wise-Contribution-Weighting/blob/main/Pipeline.png "Pipeline")

## Abstract
  Recent deep learning-based multi-view people detection (MVD) methods have shown promising results on existing
  datasets. However, current methods are mainly trained and evaluated on small, single scenes with a limited
  number of multi-view frames and fixed camera views. As a result, these methods may not be practical for
  detecting people in larger, more complex scenes with severe occlusions and camera calibration errors. 
  This paper focuses on improving multi-view people detection by developing a supervised view-wise contribution
  weighting approach that better fuses multi-camera information under large scenes. Besides, a large synthetic
  dataset is adopted to enhance the model's generalization ability and enable more practical evaluation and 
  comparison. The model's performance on new testing scenes is further improved with a simple domain adaptation
  technique. Experimental results demonstrate the effectiveness of our approach in achieving promising 
  cross-scene multi-view people detection performance.
  
## Overview
We release the PyTorch code for the MVDNet, a stable multi-view people detector with promising performance on CVCS,
CityStreet, Wildtrack, and MultiviewX datasets. 

## Video
![Video](https://github.com/zqyq/Multi-view-People-Detection-in-Large-Scenes-via-Supervised-View-wise-Contribution-Weighting/blob/main/video_short_gif.gif "video gif")

## Poster 
(All images and videos have been uploaded to this repository. Please take a look!...)
AAAI 2024 poster:
![Poster](https://github.com/zqyq/Multi-view-People-Detection-in-Large-Scenes-via-Supervised-View-wise-Contribution-Weighting/blob/main/poster_fig.png "Poster")

## Content
- [Dependencies](#dependencies)
- [Data Preparation](#Data Preparation)
- [Training](#Training)
- [Perspective transformation](#Perspective transformation)


## Dependencies
- python
- pytorch & torchvision
- numpy
- matplotlib
- pillow
- opencv-python
- kornia
- tqdm
- h5py
- argparse

## Data Preparation
In the code implementation, the root path of the four main datasets is defined as ```/mnt/data/Datasets```. Of course,
it can be changed.
When you apply the method to your datasets or other paths, the root path should look like this:
```
Datasets
|__CVCS
    |__...
|__CityStreet
    |__...
|__Wildtrack
    |__...
|__MultiviewX
    |__...
```
## Training
 During the training phase, we need to train the model in 3 stages, the feature extractor is shared across all camera views, i.e., ResNet18 and VGG16. 

Take training a detector on CVCS dataset as an example, to train the final detector, run the following script in order.
```shell script
python main.py -d cvcs --variant 2D 
```
After obtaining the trained 2D feature extractor, we set the path of the extractor as ```args.pretrain```, 
assuming it is "/trained_2D.pth". Next, we train the detector for single-view prediction.
```
python main.py -d cvcs --variant 2D_SVP --pretrain /trained_2D.pth
```
Samely, when the single-view detector is trained well, assuming it is ```/trained_2D_SVP.pth```. Next, 
we train the final detector.
```
python main.py -d cvcs --variant 2D_SVP_VCW --pretrain /trained_2D_SVP.pth
```
On Wildtrack and MultiviewX, we take the final detector trained on CVCS as the model, then test it with fine-tuning 
and domain-adaptation techniques.


## Pretrained models
You can download the checkpoints at this link.

## Acknowledgement
This work was supported in parts by NSFC (62202312, 62161146005, U21B2023, U2001206), DEGP Innovation Team 
(2022KCXTD025), CityU Strategic Research Grant (7005665), and Shenzhen Science and Technology Program 
(KQTD20210811090044003, RCJC20200714114435012, JCYJ20210324120213036).

## Reference
```
@inproceedings{MVD24,
title={Multi-view People Detection in Large Scenes via Supervised View-wise Contribution Weighting},
author={Qi Zhang and Yunfei Gong and Daijie Chen and Antoni B. Chan and Hui Huang},
booktitle={AAAI Conference on Artificial Intelligence},
pages={7242--7250},
year={2024},
}
```
