# Handwritten Number Predictor

## Install

Clone the repository and unzip the Images folder.

## Features

The app.py file is the main file that run the interface where users can draw a number from 0-9, and the model will attempt to predict it. Change the model path in app.py to switch between CNN and MLP.

The loadNumbers.py contains the dataset class used for the images as well as the model class used train and predict. The train_model.py file is the script used to train the models and the MLP.pth and CNN.pth contains the weights for the trained MLP model and trained CNN model respectively. The interface.py contains the code for the interface.

The raw image data used to train the model is stored in the Images folder, and the labels.csv contain the labels for those images. All 100,000+ images were downloaded from: https://www.kaggle.com/datasets/jcprogjava/handwritten-digits-dataset-not-in-mnist 

## MLP Model

This project includes an MLP model designed to classify flattened grayscale 28×28 images of handwritten digits (0–9). Unlike a CNN, the MLP treats each pixel independently, ignoring spatial structure, but can still achieve high accuracy on this dataset.

Model Architecture:

| Layer       | Details                                              | Output Shape |
| ----------- | ---------------------------------------------------- | ------------ |
| **Flatten** | Converts 28×28 input image into a 784-element vector | [784]        |
| **FC1**     | Fully connected, 784 → 512                           | [512]        |
| **ReLU**    | Activation function                                  | [512]        |
| **FC2**     | Fully connected, 512 → 512                           | [512]        |
| **ReLU**    | Activation                                           | [512]        |
| **FC3**     | Fully connected, 512 → 10 (logits)                   | [10]         |

The loss function used was Cross Entropy and Adam was used as the optimizer.

The accuracy of the saved model was 99.2% on a random split of 80:20 of training to test data for 10 epochs. The model was trained using 80,000+ images of handwritten digits and tested using 20,000 images. 

## CNN Model 

This project includes a CNN model designed to classify grayscale 28×28 images of handwritten digits (0–9). The network leverages spatial patterns in the image to achieve high accuracy.

Model Architecture:

| Layer        | Details                                                                       | Output Shape |
| ------------ | ----------------------------------------------------------------------------- | ------------ |
| **Conv1**    | 2D Convolution, 1 input channel → 16 output channels, kernel size=3, stride=1 | [16, 26, 26] |
| **ReLU**     | Activation function                                                           | [16, 26, 26] |
| **MaxPool1** | 2×2 max pooling                                                               | [16, 13, 13] |
| **Conv2**    | 2D Convolution, 16→32 channels, kernel size=3                                 | [32, 11, 11] |
| **ReLU**     | Activation                                                                    | [32, 11, 11] |
| **MaxPool2** | 2×2 max pooling                                                               | [32, 5, 5]   |
| **Conv3**    | 2D Convolution, 32→64 channels, kernel size=3                                 | [64, 3, 3]   |
| **ReLU**     | Activation                                                                    | [64, 3, 3]   |
| **MaxPool3** | 2×2 max pooling                                                               | [64, 1, 1]   |
| **Flatten**  | Flatten feature maps                                                          | [64]         |
| **FC1**      | Fully connected, 64→32                                                        | [32]         |
| **ReLU**     | Activation                                                                    | [32]         |
| **FC2**      | Fully connected, 32→10 (logits)                                               | [10]         |

The loss function used was Cross Entropy and Adam was used as the optimizer. The kernal used was 3, and the stride was 1. 2x2 window was used for max pooling.

Accuracy of this model was 99.9% on the same random split. 

## Limitations

The biggest limitation of the MLP is that it has a hard time differentiating 0's from 2's. It consistently will predict 2 if a 0 is clearly drawn. This can be mainly attributed to MLP models struggling with spatial limitations due to flattening the image to a 1D array. 

CNN does not have this limitation since the model is able to detect local patterns, giving it more accurate feature recognition. This can be seen as the model is able to accurately differentiate between all numbers including 0's when analyzing user inputs.
