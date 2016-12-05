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

Vist the following link to see our published ICASSP 2016 conference
paper for this project.

https://www.researchgate.net/publication/290428052_Fast_Anomaly_Detection_in_Traffic_Surveillance_Video_based_on_Robust_Sparse_Optical_Flow

# Datasets
For datasets used in paper `Fast anomaly detection in traffic surveillance video based on robust sparse optical flow`,
visit the following link:
https://drive.google.com/drive/folders/0B36U1cGE5GRiRXpsTjNKQmwtVTA?usp=sharing
or
http://pan.baidu.com/s/1pKFvlMz


+ Roadcross1, Roadcross2, Pedestrian and RainyNight are our custom dataset. They are recorded in real-life traffic
and there are no groundtruth.
+ UCSDPed1 and USCSDPed2 are originally published by UCSD and are in image formats. We make a video version of the 
two datasets and make it easy to be used in video anomaly detection.


# At Last

If you use our datasets or code in your work, please cite our paper
@inproceedings{tan2016fast,
  title={Fast anomaly detection in traffic surveillance video based on robust sparse optical flow},
  author={Tan, Hanlin and Zhai, Yongping and Liu, Yu and Zhang, Maojun},
  booktitle={2016 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={1976--1980},
  year={2016},
  organization={IEEE}
}