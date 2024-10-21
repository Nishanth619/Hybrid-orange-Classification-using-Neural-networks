# Hybrid Oranges Classification

This repository contains the implementation of a deep learning model for classifying different types of hybrid oranges. The project uses various advanced models like InceptionV3, MobileNet, and CNN for enhanced classification accuracy. The primary goal is to improve fruit quality control and optimize inventory management through automated classification.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Performance and Snapshots](#performance-and-snapshots)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Project Overview
The project focuses on building a deep learning pipeline to classify hybrid oranges into different categories using image data. The pipeline involves data preprocessing, model training, evaluation, and performance tuning to achieve high accuracy.

### Models Used
- **InceptionV3**
- **MobileNet**
- **Custom CNN**

## Dataset
The dataset used for this project can be found on Kaggle. It contains images of various hybrid oranges, classified into different types.

### Kaggle Dataset
[Hybrid Oranges Dataset on Kaggle](https://www.kaggle.com/datasets)

To use the dataset in this project:
1. Download the dataset from Kaggle.
2. Extract the files and place them in the `data/` directory of this repository.


## Model Architecture
The following models were implemented for the classification task:
1. **InceptionV3**: Pretrained on ImageNet, this model was fine-tuned with hybrid orange data for improved accuracy.
2. **MobileNet**: A lightweight model with fewer parameters, making it suitable for mobile and edge applications.
3. **Custom CNN**: A custom-built convolutional neural network designed specifically for this classification task.

## Performance and Snapshots
### Performance Metrics
- **InceptionV3**: Achieved 95% accuracy on the validation set.
- **MobileNet**: Achieved 92% accuracy on the validation set.
- **Custom CNN**: Achieved 89% accuracy on the validation set.

## Installation
To set up the project, clone the repository and install the required packages using `requirements.txt`.

## Usage
Run the training script to start training the models and evaluate their performance.

## Contributing
Contributions are welcome! Feel free to open issues or submit pull requests.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
