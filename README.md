# FoV_Prediction 基于KNN和LR的视点预测
Predict user FoV using KNN and LR

## Dataset^M
---
1. all recorded gaze data are in 'Gaze_txt_files' directory, under this directory, the data for one person ^M
is placed in one directory and the data for one video is stored in one txt file. each line in txt file ^M 
is organized as follows: "frame, frame index, forward, head position x, head position y, eye, gaze position x,^M
gaze position y", where frame index starts from 1, head position and gaze position are the position of head^M
and eye in the panorama image, they are fractional from 0.0 to 1.0 with respect to the panorama image and^M
computed from left bottom corner^M

## Requirement^M
---
1. Train and test using sliding window, train longitude and latitude separately. ^M
2. Train and acquire accuracy.^M
KNN: choose k.
LR: choose size of sliding window.
