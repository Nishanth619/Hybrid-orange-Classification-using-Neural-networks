# Hybrid Oranges Classification

This repository contains the implementation of a deep learning model for classifying different types of hybrid oranges. The project uses various advanced models like **InceptionV3**, **MobileNet**, and **CNN** for enhanced classification accuracy. The primary goal is to improve fruit quality control and optimize inventory management through automated classification.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Model Architecture](#model-architecture)
4. [Performance and Snapshots](#performance-and-snapshots)
5. [Installation](#installation)
6. [Usage](#usage)
7. [Contributing](#contributing)
8. [License](#license)

## Project Overview
The project focuses on building a deep learning pipeline to classify hybrid oranges into different categories using image data. The pipeline involves data preprocessing, model training, evaluation, and performance tuning to achieve high accuracy.

The models used for classification:
- **InceptionV3**
- **MobileNet**
- **Custom CNN**

## Dataset
The dataset used for this project can be found on **Kaggle**. It contains images of various hybrid oranges, classified into different types.

**Kaggle Dataset:** [Hybrid Oranges Dataset on Kaggle]([https://www.kaggle.com/username/hybrid-oranges-dataset](https://www.kaggle.com/datasets/jacko9812/colombian-citric-fruits)

To use the dataset in this project:
1. Download the dataset from Kaggle.
2. Extract the files and place them in the `data/` directory of this repository.

```bash
data/
├── train/
├── val/
└── test/

## Model Architecture
The following models were implemented for the classification task:

InceptionV3: Pretrained on ImageNet, this model was fine-tuned with hybrid orange data for improved accuracy.
MobileNet: A lightweight model with fewer parameters, making it suitable for mobile applications.
Custom CNN: A custom-built convolutional neural network designed specifically for this classification task.

