# Mask R-CNN for Object Detection and Segmentation in ArcGIS Pro

This is a fork of the original Mask R-CNN by matterport, see https://github.com/matterport/Mask_RCNN for installation, requirements etc..
Some additional scripts are added here which I used to train, evaluate and convert output from ArcGIS to use with matterport's version of Mask R-CNN

# Steps to do deep learning object detection in ArcGIS Pro

This is how I got the tool 'Detect objects using deep learning' in ArcGIS Pro to work using houses as an example.

## 1. Create masks over the objects

![](assets/detection_anchors.png)