# Faster-R-CNN-YOLO-V3
Training and Testing Object Detection Models Faster R-CNN and YOLO V3 on the VOC Dataset

# Object Detection on VOC Dataset

## Overview

This repository contains the implementation of Faster R-CNN and YOLO V3 models trained on the VOC dataset using the MMDetection framework.

## Setup

1. Clone the repository and navigate to the directory:
    ```bash
    git clone https://github.com/your-repo.git
    cd your-repo
    ```

2. Install dependencies:
    ```bash
    pip install -U torch==1.13.1+cu121 torchvision==0.14.1+cu121 torchaudio==0.13.1+cu121 -f https://download.pytorch.org/whl/cu121/torch_stable.html
    pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch1.13.0/index.html
    pip install mmdet==2.28.2
    pip install matplotlib opencv-python
    ```

3. Download and extract the VOC dataset:
    ```bash
    mkdir -p /content/data/VOCdevkit
    cd /content/data/VOCdevkit
    wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
    wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
    wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
    tar -xvf VOCtrainval_06-Nov-2007.tar
    tar -xvf VOCtest_06-Nov-2007.tar
    tar -xvf VOCtrainval_11-May-2012.tar
    cd /content
    ```

## Training

1. Train Faster R-CNN:
    ```bash
    python train_faster_rcnn.py
    ```

2. Train YOLO V3:
    ```bash
    python train_yolov3.py
    ```

## Testing and Visualization

1. Test and visualize the models:
    ```bash
    python test_and_visualize.py
    ```

## Tensorboard Visualization

1. Run Tensorboard to visualize training logs:
    ```bash
    %load_ext tensorboard
    %tensorboard --logdir /content/work_dirs/
    ```

## Trained Model Weights

Download the trained model weights from the following links:
- [Faster R-CNN](https://drive.google.com/your-faster-rcnn-model)
- [YOLO V3](https://drive.google.com/your-yolo-v3-model)
