# PoseNet re-implementation

A reimplementation of ["PoseNet: A Convolutional Network for Real-Time 6-DOF Camera Relocalization."](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Kendall_PoseNet_A_Convolutional_ICCV_2015_paper.pdf) by Alex Kendall, Matthew Grimes and Roberto Cipolla.

This implementation differs slightly in that I use a residual network instead of GoogLeNet. I also include a script to turn a rosbag into a TF record dataset.
