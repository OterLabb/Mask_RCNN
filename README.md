# Mask R-CNN for Object Detection and Segmentation in ArcGIS Pro

This is a fork of the original Mask R-CNN by matterport, see https://github.com/matterport/Mask_RCNN for installation, requirements etc..
Some additional scripts are added here which I used to train, evaluate and convert output from ArcGIS to use with matterport's version of Mask R-CNN

# Steps to do deep learning object detection in ArcGIS Pro

This is how I got the tool 'Detect objects using deep learning' in ArcGIS Pro to work using houses as an example.

## 1. Create masks over the objects of interest
Using the training samples manager, create masks over as many objects as possible, if your model is performing poorly, think about adding more training samples.
![](assets/training_samples.png)

## 2. Export training data
Export the training data, using the tool 'Export training data for deep learning', I like to use tiff format, as it seems to perform slightly better then compressed jpg's. Set the tile and stride size, or leave them as default. Choose RCNN_Masks as the meta data format and export the tiles.
![](assets/export_samples.png)
