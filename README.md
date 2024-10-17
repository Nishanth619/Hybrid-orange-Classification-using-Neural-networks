# Hybrid Oranges Classification

This project is focused on classifying different types of hybrid oranges using deep learning models, including InceptionV3 and MobileNet. The goal is to enhance quality control and improve overall fruit quality in agricultural sectors.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Project Overview
This project implements a deep learning-based solution to classify hybrid oranges. The dataset includes 8 classes of citrus fruits, including hybrid varieties. The models have been trained using **InceptionV3** and **MobileNet** architectures with fine-tuning for improved accuracy.

## Dataset
The dataset is divided into training, validation, and test sets:
- **Training Set**: 80%
- **Validation Set**: 20%

## Model Architecture
We explored multiple architectures for classification, including:
- **InceptionV3**: Fine-tuned for 8 classes.
- **MobileNet**: Lightweight architecture optimized for mobile and embedded devices.

## Results
- InceptionV3 Accuracy: 92%
- MobileNet Accuracy: 89%

## Setup and Installation

### Requirements:
- Python 3.x
- TensorFlow
- Keras
- OpenCV
- NumPy
- Matplotlib

Install the required dependencies by running:
```bash
pip install -r requirements.txt
