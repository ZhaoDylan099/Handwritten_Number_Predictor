# Handwritten Number Predictor

## Install

Clone the repository and unzip the Images folder.

## Features

The app.py file is the main file that run the interface where users can draw a number from 0-9, and the model will attempt to predict it.

The loadNumbers.py contains the dataset class used for the images as well as the model class used train and predict. The train_model.py file is the script used to train the model and the model.pth contains the weights for the trained model. The interface.py contains the code for the interface.

The raw image data used to train the model is stored in the Images folder, and the labels.csv contain the labels for those images.

## Model

The model designed is a simple 3-layer neural network with ReLU activation. 

Input
  |
  v
Linear - ReLU (512 neurons)
  |
  v
Linear - ReLu (512 neurons)
  |
  v
Linear (output) (10 neurons)

The accuracy of the saved model was 99.7% on a random split of 80:20 of training to test data. The model was trained using 80,000+ images of handwritten digits downloaded from https://www.kaggle.com/datasets/jcprogjava/handwritten-digits-dataset-not-in-mnist and tested it using 20,000 images.

## Limitations

The biggest limitation of this model is that it has a hard time differentiating 0's from 2's. It consistently will predict 2 if a 0 is clearly drawn. I am not sure exactly why this occurs but I believe is that model receives the image in only one oreintation so if the 0's are not exactly shaped as the training samples, it may not properly recognize it.
