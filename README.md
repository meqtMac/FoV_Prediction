# FoV_Prediction 基于KNN和LR的视点预测
Predict user FoV using KNN and LR

## Dataset
---
All recorded gaze data are in 'Gaze_txt_files' directory, under this directory, the data for one person is placed in one directory and the data for one video is stored in one txt file. each line in txt file is organized as follows: "frame, frame index, forward, head position x, head position y, eye, gaze position x, gaze position y", where frame index starts from 1, head position and gaze position are the position of head and eye in the panorama image, they are fractional from 0.0 to 1.0 with respect to the panorama image and computed from left bottom corner.

## Requirement
---
1. Train and test using sliding window, train longitude and latitude separately.
2. Train and acquire accuracy.
KNN: choose k.
LR: choose size of sliding window.
