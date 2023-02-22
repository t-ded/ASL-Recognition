# -*- coding: utf-8 -*-
# disable=C0301, C0103, E1101
"""
A simple function to enable running
the different scripts from command line.

Parameters:

    collect: string
        Start the process of data collection.
    train: string
        Train the convolutional neural network.
    showcase: string
        Starts the camera and showcases the model prediction using pretrained model.

Default procedure (when no parameters given) is trying to run the whole process
    That is folders setup, data collection (default amount 500),
    CNN build (default 2 layers) and train and then model showcasing
"""

import argparse
import os
import tensorflow as tf
import json
import utils
from collect_dataset import collect_data
from showcase_collect_preprocessing import showcase_preprocessing

parser = argparse.ArgumentParser()

# Specify procedure
parser.add_argument("--c", "--collect", action="store_true", help="If given, run the data collection process")
parser.add_argument("--p", "--preprocess", action="store_true", help="If given, showcase various preprocessing pipelines")
parser.add_argument("--t", "--train", action="store_true", help="If given, run the model training process")
parser.add_argument("--s", "--showcase", action="count", default=0, help="If given, run the showcasing process (give twice to run with model prediction)")
parser.add_argument("--config_dir", default="", type="str", help="Directory with the config.json file")

# Specify the parameters for this experiment
# TODO:
    # Give options either to:
        # specify hyperparameters (and possibly model architecture) with some default values
            # then create new folders for this experiment and save the layout there
        # give a directory / experiment number with pre-made json file


def main(args):
    """Command line function"""

    # Load configuration file from json in the given folder
    with open(args.config_dir, "r") as config_file:
        config = json.load(config_file)

    # Default procedure
    if not args:
        print("No arguments specified, will try to run the showcasing without prediction")
        args.showcase = 1

    # Set up the list of gestures
    with open(config["Paths"]["Gesture list"], "r") as gesture_list:
        gestures = gesture_list.readlines().split(", ")

    # Set up folders and necessary variables
    data_dir, example_dir, desired_amount, current_amount, paths = utils.setup_folders(script_directory=os.path.dirname("run.py"),
                                                                                       gestures_list=gestures,
                                                                                       amount_per_gesture=config["General parameters"]["Desired amount"])
    print("The folders have been set up.")

    # Collection of the data
    if args.collect:

        print("Starting the data collection process.")

        # If the run.py is executed with desired amount already filled, the
        # user has an option to top-up the desired amount by a bit
        if all(cur >= des for des, cur in zip(desired_amount.values(),
                                              current_amount.values())):
            increment = config["General parameters"]["Top-up amount"]
            print("You are trying to run the collection procedure even though",
                  "the desired amount has already been reached.",
                  "Do you want to increase the desired amount per gesture",
                  f"by {increment}?")
            if input("Proceed (y/[n])?").lower() == "y":
                desired_amount = {letter: desired_amount[letter] + increment for letter in desired_amount.keys()}

        # Perform the dataset collection procedure
        collect_data(gestures, examples=example_dir,
                     data_directory=data_dir, current_amounts=current_amount,
                     desired_amounts=desired_amount, gesture_paths=paths,
                     translations=config["Paths"]["Translations"],
                     img_size=config["General parameters"]["Image size"])
        print("Your data has been collected, please check the folders.")

    # Showcase various preprocessing pipelines and enable saving their outputs
    if args.preprocess:
        showcase_preprocessing()

    # Build a new model and train it on the given data
    if args.train:
        # !!! TODO !!!

        # Preprocessing stage of the model
        # TODO: create and import model.py script with a functionality
        # for model building
        # This function will have an argument for preprocessing vs training
        # and will either load both of these models from
        # given files with weights, build the models from json files with
        # architecture or create a new model prototype
        pass

        # Trainable stage of the model
        # TODO: Use the model building function from the previous TODO

        # Loading the training and testing datasets from directories and optimizing them for performance
        _ = """
        model = CNN.build_model(labels=len(gestures))
        model.compile(optimizer="adam",
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=["accuracy"])
        train_images = tf.keras.preprocessing.image_dataset_from_directory(data_dir,
                                                                           validation_split=0.25,
                                                                           subset="training",
                                                                           seed=123,
                                                                           image_size=(196, 196),
                                                                           color_mode="grayscale")
        test_images = tf.keras.preprocessing.image_dataset_from_directory(data_dir,
                                                                          validation_split=0.25,
                                                                          subset="validation",
                                                                          seed=123,
                                                                          image_size=(196, 196),
                                                                          color_mode="grayscale")
        model.fit(train_images, batch_size=20, epochs=10, validation_data=(test_images), steps_per_epoch=100)
        print("Model has been built, showing model summary now.")
        print(model.summary())
        model.save_weights("Weights/weights")"""

    # Demonstrate the image taking process and possibly showcase the model and its predictions
    if args.showcase:
        # !!! TODO !!!

        # TODO: Evaluate whether args.showcase is > 1 (prediction) or not (plain showcasing)
        _ = """
        data_dir, example_dir, desired_amount, current_amount, paths = image_collection.setup_folders(os.path.dirname("image_collection.py"), gestures, 100)
        if "train" in argv:
            image_collection.image_capturing(gestures, examples=example_dir, save=False, predict=True, data_directory=data_dir, model=model)
        else:
            model = CNN.build_model(labels=len(gestures), input_shape=(196, 196))
            try:
                model.load_weights("Weights/weights").expect_partial()
            except Exception as exception:
                if exception.__class__.__name__ == "NotFoundError":
                    print("Please either use the train command along with a showcase command",
                          "or insert a model weights folder in the directory of this script.")
                    print("The program will now terminate")
                return
        image_collection.image_capturing(gestures, examples=example_dir, save=False, predict=True, data_directory=data_dir, model=model)"""


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
