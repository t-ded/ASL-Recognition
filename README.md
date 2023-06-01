# Recognition of the American Sign Language in real time

## Description
This project is a real-time American Sign Language (ASL) recognition system built using mainly TensorFlow, Keras, OpenCV2 and Python. It features multiple scripts for tasks ranging from data collection over model building and training to real-time model prediction from camera. The project was built as a fundamental component of the bachelor's thesis of mine, with the topic of "Construction of a Neural Networks model for translation of recorded sign language".

The repository also features a model pre-trained on a dataset collected by myself that features around 2,000 samples per each of 49 unique categories. The dataset comprises hands of 10 different people aged 10-64 of both genders that used both their hands while signing the gestures in front of almost 50 unique backgrounds under various lighting conditions. This helped the model achieve substantial ability to generalize to previously unseen data and reach real-time accuracy of 100 % for easier backgrounds with any lighting conditions and 93 % for extremely hard backgrounds with very poor lighting conditions.

## Table of Contents

- [Description](#description)
- [Features](#features)
- [Demonstration](#demonstration)
- [Installation](#installation)
- [Components and Project Structure](#components-and-project-structure)
- [Usage](#usage)
  - [Environment Demonstration](#enviornment-demonstration)
  - [Data Collection](#data-collection)
  - [Model Building and Training](#model-building-and-training)
  - [Preprocessing Pipelines Demonstration](#preprocessing-pipelines-demonstration)
  - [Real-time Model Deployment](#real-time-model-deployment)
- [Common Issues and How to Fix Them](#common-isues)
  - [ImportError: cannot import name 'builder' from 'google.protobuf.internal'](#importerrror)

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

!!!!

<span style="color:red">*TODO*</span>.

!!!!

## Installation

This section shall guide the full installation process. Presence of Python v3.10.9 and pip package installer on the device and basic knowledge of Python is assumed in this section and generally throughout this whole repository.

The first step is to clone this repository, fork it or download ZIP version of the code. As the next step, setting up a virtual environment for Python is strongly recommended. All the necessary packages can then be installed from the requirements.txt file, which is a part of the repository. Another way is to set up a virtual environment with all the requirements via Conda (package manager). These steps can look along the following lines:

```
# To create a virtual environment with all required packages via conda:
conda env create -f requirements.yml

# To install all packages specified in the requirements.txt file via pip:
pip install -r requirements.txt
```

## Components and Project Structure

Firstly, a slight disclaimer - please make sure to preserve the layout of the project as it is, since changing the folders, filenames or location of the files may lead to unexpected behaviour. Also note that the main parts of this script will only work on a device with a working camera. Also, the scripts utilizing image capturing might need slight tweaking in case the camera cannot be found in the default spot by `opencv2` (see [Common Issues and How to Fix Them](#common-issues) at the bottom of this guide).

As it currently stands, the root directory of this project contains 6 folders and numerous files. This section shall provide a brief summary for most of these with further breakdown for the most important functionalities being presented in the following [Usage](#usage) section.

* `pretrained_log` provides a TensorBoard log directory for the pre-trained model.
* `Data` folder contains a small sample dataset, which is insufficient for satisfactory live recognition from camera but is enough for presentation of the training process.
* `Examples` folder stores the example images for each of the gestures in the dataset. These instruct the users on how to perform each individual sign.
* `preprocessing_pipelines` is the folder to store images of various preprocessing pipelines for comparison. This is where images from `showcase_collect_preprocessing.py` are saved.
* `model` stores the scripts regarding model training as well as the models themselves.
  * `model/current` folder is where the current pre-trained model is saved.
  * `model/experiments` folder is created to store subsequent model training runs when starting training for the first time (not originally present in this repository).
  * `model/knn_summary.txt` and `model/rfc_summary.txt` provide evaluations of the classical ML models used in this research for comparison (k-Nearest Neighbours and Random Forest).
  * `model/model.py` is a script for building the model and preprocessing pipeline based on given text instructions.
  * `model/preprocessing.py` is a script containing auxiliary classes and functions, such as custom TensorFlow layers and callbacks.
* `collect_dataset.py` is a script that takes the user through the image collection process.
* `config.json` is a configuration file for this project. Some important settings can be specified in this file, including paths to some components or the image size for the dataset.
* `gestures.txt` is a list of gestures that are used in this project.
* `run.py` is the run script for this project. Through this script, the usery can access every other component of the project by specifying the procedure in the command line.
* `showcase_collect_preprocessing.py` is a script for real-time demonstration, comparison and saving of various preprocessing pipelines.
* `showcase_model` is a script that starts the real-time environment and utilizes the model for real-time prediction if prompted to do so.
* `translations.txt` is a file with English-Czech gesture pairs to enable using the application in both languages.
* `utils.py` is a script storing various utility functions, such as functions for project initialization etc.

## Usage

This section will cover the guidelines to using all components of this project as well as all features assigned to them. From perspective of the user, every component can be accessed via the run script stored in the root directory of this project:

```
python run.py
```

From the most superficial perspective, the run script provides 5 different options:
1. `python run.py -col` or `python run.py --collect` - this starts the image collection process.
2. `python run.py -show` or `python run.py --showcase` - this is a basic procedure that can be used to familiarize the user with the environment and its controls. It starts the real-time camera view as well as shows the Example images. The user can use the same controls as with other components (see section [Real-time Prediction](#real-time-prediction) below). No images are saved and no predictions are done during this procedure. This is also the default procedure that is run when only `python run.py` is given.
3. `python run.py -prep` or `python run.py --preprocessing` - this starts the process for comparison of various preprocessing pipelines in real-time.
4. `python run.py -tr` or `python run.py --train` - this command starts the training procedure.
5. `python run.py -pred` or `python run.py --predict` - when this command is entered, the run script tries to find a pre-trained model and use it for real-time prediction.

Some of these procedures as well as their individual settings or parameters can then be further specified by additional commands. These can be found in the respective sections below.

### Image Collection

The image collection scripts starts with initializing a real-time camera view as well as a window with example gestures from the `Example` folder serving as templates for the user. The images are saved into the `Data` folder.

In the `config.json` file, the user can specify the image collection process beforehand. Apart from specifying the paths to the `translations.txt` and `gestures.txt` files, the user can set the *Image size* for the saved images (please note that the pre-trained model has been trained with image size 196, thus changing this parameter would require creating a new model for real-time inference purposes) as well as what is the *Desired amount*, that should be present in the `Data` folder per gesture at the end of the image collection procedure. For cases when the `Data` folder already contains this many files for each of the gestures, the user is prompted to confirm, whether they want to increase this amount by the *Top-up amount* (which results in collectiong Top-up amount-many images per gesture).

The real-time camera view contains a rectangle which signalizes what part of the frame is being saved as well as name of the gesture that is currently being collected, the progress in terms of the number of the current gesture and the estimated time until the next gesture (ETA) in seconds. Every time a new gesture is being collected, a larger frame with the example photo is displayed for the user to get familiar with the gesture. Then, there is a slight warm-up period, during which the images are not being saved - this is signalized by the colours of the frame and texts being red. When these turn green, the images are being saved until the desired amount of frames per gesture are obtained or until the user decides to skip to the next gesture.

To navigate the image collection process, the user has multiple controls (note that on most devices, the user must first select the live view in order for these to work):
* Pressing `Spacebar` moves the rectangle into another position (there are 3 in total - starting with lower mid, then moving to upper left and then upper right).
* `q` key can be used to end the process for the current gesture and skip to the next one.
* `l` key switches the language of the gestures.
* `p` pauses the process (and then `p` starts it again).
* `Esc` key can be used to terminate the whole process.

### Preprocessing Pipelines Comparison

Running the script for preprocessing demonstration starts the classical real-time camera view but also opens 6 other smaller windows. Each of these displays a different preprocessing pipeline working in real time. These pipelines are specified in the `config.json` file in the form of textual instructions for the `build_preprocessing` function - for guidelines on these, please refer to the [Model Building](#model-building) section.

This script offers controls in form of the `Esc` and `Spacebar` keys same as the image collection script. Furthemore, the `q` key can be used to save the current layout - in the `preprocessing_pipelines` folder, a new folder is created. This folder stores the summaries of the currently compared preprocessing pipelines and then with each press of the `q` key, a folder that contains an image for each of these.

### Model Training

### Real-time Model Prediction

## Model Building

### Preprocessing Pipelines

!!!
TODO: Describe how to build the preprocessing pipeline
!!!

### Trainable Model

## Common Issues and How to Fix Them

This chapter shall list some of the commonly observed issues that may arise when trying to implement the program with new devices as well ways found to fix them. In case of finding an issue you were not able to solve that is not presented here, please do not hesitate to contact me.

### ImportError: cannot import name 'builder' from 'google.protobuf.internal'

Solution to this issue was taken from StackOverflow answer of user user19266443 (https://stackoverflow.com/a/72494013). The error may be encountered after installing the dependencies and trying to run the run script. When encountering this error, follow these steps:
1. Install the latest protobuf version.
```
pip install --upgrade protobuf
```
2. Copy builder.py from .../Lib/site-packages/google/protobuf/internal (where .../ corresponds to the location of the project's folder or the virtual environment on your device) to another place on your computer.
3. Install a protobuf version 3.19.6.
```
pip install protobuf==3.19.6
```
4. Copy builder.py from location in step 2 to Lib/site-packages/google/protobuf/internal.
5. The code should now work.
