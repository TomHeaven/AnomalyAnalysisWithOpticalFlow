# AnomalyAnalysisWithOpticalFlow
This project is designed for anomaly detection of video sequence.
We improve the efficiency of optical flow computation with foreground
mask and spacial sampling and increase the robustness of optical flow
with good feature (TK) points selecting and forward-backward filtering.
A foreground channel is also added to the feature vector to help detect
static or low speed objects. The algorithm is validated on real-life traffic
surveillance to prove its effectiveness. It is also evaluated on a
benchmark dataset and achieve detection results comparable to
state-of-art methods and outperforms them at pixel-level when the false
alarm rate is low. The strength of our algorithm is that it runs
real-time on the benchmark dataset which is hundreds of times faster
than comparative methods.

Visit the following link to see our published ICASSP 2016 conference
paper for this project.

https://www.researchgate.net/publication/290428052_Fast_Anomaly_Detection_in_Traffic_Surveillance_Video_based_on_Robust_Sparse_Optical_Flow

# Datasets
For datasets used in paper `Fast anomaly detection in traffic surveillance video based on robust sparse optical flow`,
visit the following link:
https://drive.google.com/drive/folders/0B36U1cGE5GRiRXpsTjNKQmwtVTA?usp=sharing
or
http://pan.baidu.com/s/1pKFvlMz

Four our custom datasets and two public datasets are available via the above links:

+ Roadcross1, Roadcross2, Pedestrian and RainyNight are our custom dataset. They are recorded in real-life traffic
and there are no groundtruth.
+ UCSDPed1 and USCSDPed2 are originally published by UCSD and are in image formats. We make a video version of the 
two datasets and make it easy to be used in video anomaly detection.

# Compiling

The project is developed using Visual Studio 2013 and OpenCV 2.4.x. You need to configure local OpenCV path to compile the project, which will generate a binary file ``AnomalyAnalysisWithOpticalFlow.exe``.

# Usage

All parameters are:
```
AnomalyAnalysisWithOpticalFlow.exe videoFolder trainVideo testVideo [1-4] thresh [test | train] BLOCK_WIDTH BLOCK_HEIGHT
```
We are going to explain them one by one.


## Run the whole pipepline:
Suppose ``train.avi`` and ``test.avi`` are stored in folder ``E:\videos``, run the following command in cmd:

```
AnomalyAnalysisWithOpticalFlow.exe E:\videos train.avi test.avi
```
This will process videos with default parameters.

## Run a stage:
The pipeline consist of four stage:

+ 1: background substraction for training and test videos
+ 2: optical flow computation for training and test videos
+ 3: abnormal detection
+ 4: generate result video by combining input video with binary abnormal detection result video

Those four stages can be run seperately to save time for debugging. For example:

If you simply need to extract background video, run

```
AnomalyAnalysisWithOpticalFlow.exe E:\videos train.avi test.avi 1
```

If the background videos have been generated, you can directly run stage 2 by:

```
AnomalyAnalysisWithOpticalFlow.exe E:\videos train.avi test.avi 2
```

Similarly, if the first two stages has been run, you can adjust threshold by directly run stage 3:

```
AnomalyAnalysisWithOpticalFlow.exe E:\videos train.avi test.avi 3 0.02
```
where ``0.02`` is the threshold to detect abnormal events.

And generate result video with

```
AnomalyAnalysisWithOpticalFlow.exe E:\videos train.avi test.avi 4
```

## Switch test video

If you need to switch test video while keeping the training video unchanged, the fastest way is to run:

```
AnomalyAnalysisWithOpticalFlow.exe E:\videos train.avi test.avi 1 0.02 test
AnomalyAnalysisWithOpticalFlow.exe E:\videos train.avi test.avi 2 0.02 test
AnomalyAnalysisWithOpticalFlow.exe E:\videos train.avi test.avi 3 0.02
AnomalyAnalysisWithOpticalFlow.exe E:\videos train.avi test.avi 4 0.02
```

## Change block size
The algorithm is block/patch based. The default block size is ``16x16``. You can change the block size to ``24x24`` by adding the last two parameters

```
AnomalyAnalysisWithOpticalFlow.exe E:\videos train.avi test.avi 1 0.02 test 24  24
AnomalyAnalysisWithOpticalFlow.exe E:\videos train.avi test.avi 2 0.02 test 24  24
AnomalyAnalysisWithOpticalFlow.exe E:\videos train.avi test.avi 3 0.02 24  24
AnomalyAnalysisWithOpticalFlow.exe E:\videos train.avi test.avi 4 0.02 24  24
```

## Switch between CPU and GPU

Using GPU (Nvidia CUDA) is enabled by default since it speeds up computation, which requires OpenCV to be compiled with CUDA support. If you do not have an Nvidia GPU or OpenCV with CUDA support, you can disable this feature by comment out the

```
#define USE_CUDA
```
in stdafx.h and recompile the project. 



# Reference

If you use our datasets or code in your work, please cite our paper

```
@inproceedings{tan2016fast,
  title={Fast anomaly detection in traffic surveillance video based on robust sparse optical flow},
  author={Tan, Hanlin and Zhai, Yongping and Liu, Yu and Zhang, Maojun},
  booktitle={2016 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={1976--1980},
  year={2016},
  organization={IEEE}
}
```