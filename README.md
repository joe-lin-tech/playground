# Project Playground
> A collection of machine learning projects in computer vision and natural language processing.

## Overview
- [image](https://github.com/joe-lin-tech/playground/tree/main/image) - a barebones convolutional neural network trained on CIFAR-10 for **image classification**
- [emotion](https://github.com/joe-lin-tech/playground/tree/main/emotion) - a fine-tuned SqueezeBERT model trained on Google's GoEmotions dataset and a HuggingFace emotion dataset for **text emotion analysis**
- [translate](https://github.com/joe-lin-tech/playground/tree/main/translate) - a basic transformer trained on a translation dataset for **sequence to sequence translation**
- [yolo](https://github.com/joe-lin-tech/playground/tree/main/yolo) - a YOLO (You Only Look Once) model pretrained on ImageNet and trained on PASCAL's VOC dataset for **object detection**

## Project Structure
Each project contains core preprocessing, training, and inference scripts, which are described below.

```params.py``` defines high-level project constants and model hyperparameters.

```dataset.py``` defines a custom PyTorch ```Dataset``` class definition for the dataset used and includes relevant data preprocessing methods.

```model.py``` defines a custom PyTorch ```Module``` class definition for the model architecture used.

```train.py``` contains the main training loop logic.

```predict.py``` contains the prediction script for model inference.