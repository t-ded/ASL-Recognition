# Recognition of the American Sign Language in real time

## Description
This project is a real-time American Sign Language (ASL) recognition system built using mainly TensorFlow, Keras, OpenCV2 and Python. It features multiple scripts for tasks ranging from data collection over model building and training to real-time model prediction from camera. The project was built as a fundamental component of the bachelor's thesis of mine, with the topic of "Construction of a Neural Networks model for translation of recorded sign language".

The repository also features a model pre-trained on a dataset collected by myself that features around 2,000 samples per each of 49 unique categories. The dataset comprises hands of 10 different people aged 10-64 of both genders that used both their hands while signing the gestures in front of almost 50 unique backgrounds under various lighting conditions. This helped the model achieve substantial ability to generalize to previously unseen data and reach real-time accuracy of 100 % for easier backgrounds with any lighting conditions and 93 % for extremely hard backgrounds with very poor lighting conditions.

## Table of Contents

- [Description](#description)
- [Features](#features)
- [Demonstration](#demonstration)
- [Installation](#installation)
- [Usage](#usage)
  - [Environment Demonstration](#enviornment-demonstration)
  - [Data Collection](#data-collection)
  - [Model Building and Training](#model-building-and-training)
  - [Preprocessing Pipelines Demonstration](#preprocessing-pipelines-demonstration)
  - [Real-time Model Deployment](#real-time-model-deployment)

## Features

* Live camera environment with handmade examples illustrating gestures in the dataset
* Image collection environment for navigating the user through the data collection process
* Demonstration and comparison of various preprocessing pipelines in real time
* Convenient model and preprocessing pipeline building with textual instructions for architectures
* Model training with user friendly hyperparameter specification
* Model deployment for real-time prediction with the device's camera
* Model prediction voicing

## Demonstration

This video features a short demonstration of real-time predictions for the model:
<span style="color:red">TODO</span>.

## Installation

This section shall guide the full installation process. Presence of Python v3.10.9 and pip v22.3.1 package installer on the device and basic knowledge of Python is assumed in this section and generally throughout this whole repository.

The first step is to clone this repository. As the next step, setting up a virtual environment for Python is strongly recommended.
<span style="color:red">TODO: Add a minimal working requirements.txt file and note how to set up venv from the requirements.txt!</span>.

The `Data` folder contains a small sample dataset, which is insufficient for satisfactory live recognition from camera but is enough for presentation of the training process.

Please make sure to preserve the layour of the project as it is, since changing the folders or location of the files may lead to unexpected behaviour.  
Also note that the main parts of this script will only work on a device with a working camera. The script for image capturing might need slight tweaking in case the camera cannot be found in the default spot by `opencv2`.  

## Usage

This section will cover the guidelines to using all components of this project as well as all features assigned to them. From perspective of the user, every component can be accessed via the run script stored in the root directory of this project:

```
python run.py
```

Further command line arguments then specify the distinct procedures as well as their respective settings and parameters.

The folder contains three key Python programs.  
The `CNN.py` contains a `build_model` function, which enables the user to build a very basic convolutional neural network with a given number of layers.  
The `image_collection.py` is the core module which contains multiple functions used for the purpose of setting up the project, folders and then capturing the images.  
Last but not least, the `run.py` is the run script for this project.  



There are three supported commands for this function:  

1. collect - sets the project up and starts the collection process (opens the camera and starts capturing and saving the images for the dataset). Note that by default, the expected dataset size is 500 samples per gestures, which is the size of the sample dataset, thus this script will terminate immediately after execution.  
2. train - use the images from the `Data` folder for training a simple convolutional neural network (by default 2 convolutional, 1 pooling and 2 dense layers) and saving the weights to `Weights` folder.  
3. showcase - run the image capturing script and without saving the images, run live prediction using the model loaded from `Weights` folder or the previously trained model if `showcase` has been used alongside `train` parameter. The model does not perform well on live prediction with such insufficient dataset.  

An example usecase is as follows:  

```
python run.py train showcase
```

When the image capturing script is being ran, you will see 4 windows.   
In the left, there is your live view alongside with a rectangle and predicted sign in case of prediction being expected. You can use `spacebar` key to move the rectangle into one of three positions for better comfortability. You can also use the `Esc` key to terminate the script.  
In the middle, the upper window shows grayscaled version and the lower window shows the binarized form of the rectangle in the left window - this is the part of the image on which the prediction is being made and the part that is saved during the dataset collection process.  
In the right window, you can see the author of this project performing example gestures. During dataset collection process, this is the expected gesture, during showcasing, these serve as an inspiration for the user. You can move to the following gesture (during both processes) using the `q` key.   

Apart from the scripts, the project also contains tests for all of the functions in the project.  
These can be ran using the following commands in the command line:  

```
python -m pytest test_CNN.py
python -m pytest test_image_collection.py
```

## Important notice

Please do not share the repository since it contains code for an academic paper as well as personal data.  

## Dataset collection

Since the main goal of this project at the moment is to build a dataset, help with this goal is appreciated. If you would like to contribute to this process, do not hesitate to contact me.  
The `collect` argument is designed for this purpose - note that by default, the expected size is set to 500 images per gesture, which is the size of the sample dataset in the `Data` folder. This is why the process terminates immediately when executed.  
