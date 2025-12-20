# Handwritten Number Predictor

## Install

Clone the repository and unzip the Images folder.

## Features

The app.py file is the main file that run the interface where users can draw a number from 0-9, and the model will attempt to predict it.

The loadNumbers.py contains the dataset class used for the images as well as the model class used train and predict. The train_model.py file is the script used to train the model and the model.pth contains the weights for the trained model. The interface.py contains the code for the interface.

The raw image data used to train the model is stored in the Images folder, and the labels.csv contain the labels for those images.
