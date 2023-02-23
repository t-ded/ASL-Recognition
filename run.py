# -*- coding: utf-8 -*-
# disable=C0301, C0103, E1101
# TODO: Adjust README.md file and the docstring for this run function
# TODO: Fill in the docstring and comments for classes and their methods in preprocessing.py file
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
from showcase_model import showcase_model
from preprocessing import AdaptiveThresholding, Blurring

parser = argparse.ArgumentParser()
parser.add_argument("--config_dir", default="", type=str, help="Directory with the config.json file")
parser.add_argument("--experiment", default=None, type=int, help="Number of this experiment (the settings will be saved in the respective newly created folder or loaded from an existing folder)")

# Specify procedure
procedure = parser.add_mutually_exclusive_group()
procedure.add_argument("--col", "--collect", action="store_true", help="If given, run the data collection process")
procedure.add_argument("--prep", "--preprocess", action="store_true", help="If given, showcase various preprocessing pipelines")
procedure.add_argument("--tr", "--train", action="store_true", help="If given, run the model training process")
procedure.add_argument("--show", "--showcase", action="store_true", help="If given, run the showcasing process")
procedure.add_argument("--pred", "--predict", action="store_true", help="If given, run the showcasing process with model prediction")

# Specify the hyperparameters if the json file was not given
hyperparameters = parser.add_argument_group("Hyperparameters")
hyperparameters.add_argument("--batch_size", default=64, type=int, help="Batch size")
hyperparameters.add_argument("--epochs", default=10, type=int, help="Number of epochs")
hyperparameters.add_argument("--optimizer", default="Adam", choices=["Adam", "SGD"], help="Optimizer for training")
hyperparameters.add_argument("--learning_rate", default=0.01, type=float, help="Starting learning rate")
hyperparameters.add_argument("--regularization", default=None, choices=["l1", "l2"], help="Regularization for the loss function")
hyperparameters.add_argument("--dropout", default=0.5, type=float, help="Dropout rate for the dropout layers")
hyperparameters.add_argument("--seed", default=123, type=int, help="Random seed for operations including randomness (e.g. shuffling)")

# TODO: If train argument given and experiment not specified, inform the user and change the experiment number to the last folder number + 1 in list of experiments
# TODO: If train argument given and experiment == -1, then rewrite the current model (Make sure to get the input from the user to proceed with this if some model already present)
# TODO: Add options for architecture (probably using action="append" and then expecting input such as "icccpdo" for input, conv, conv, conv, pool, dense, output)
# TODO: Create a model building function for the expected input specified in the previous todo
# TODO: Create the same functionality for preprocessing?
# TODO: Think of ways of specifying the parameters for the individual layers


def main(args):
    """Command line function"""

    # Load configuration file from json in the given folder
    with open(args.config_dir, "r") as config_file:
        config = json.load(config_file)

    # Default procedure
    if not args:
        print("No arguments specified, will try to run the showcasing without prediction")
        args.showcase = True

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
    elif args.preprocess:
        showcase_preprocessing()

    # Build a new model and train it on the given data
    elif args.train:
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

        # TODO: Create checkpoints for training and save the training from there
        # TODO: Save the model into experiment folder with given number or
        # try saving it as current if experiment == -1
        # (prompt the user for proceeding confirmation in case a model already exists)

        # Loading the training and testing datasets from directories and optimizing them for performance
        train_images, test_images = tf.keras.preprocessing.image_dataset_from_directory(data_dir,
                                                                                        validation_split=config["General parameters"]["Validation split"],
                                                                                        subset="both",
                                                                                        seed=args.seed,
                                                                                        image_size=(config["General parameters"]["Image size"],
                                                                                                    config["General parameters"]["Image size"]))
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

    # Demonstrate the image taking process
    elif args.showcase:

        showcase_model(gestures, examples=example_dir,
                       predict=False, model=None,
                       translations=config["Paths"]["Translations"],
                       img_size=config["General parameters"]["Image size"])

    # Demonstrate the image taking process while also demonstrating the model and its predictions
    elif args.predict:

        try:
            model = tf.keras.models.load_model(filepath=config["Model"]["Current model"],
                                               custom_objects={"AdaptiveThresholding": AdaptiveThresholding,
                                                               "Blurring": Blurring})
        except IOError:
            print("The prediction procedure was chosen but model cannot be found",
                  f"in the folder specified in the config.json file ({config['Model']['Current model']}).",
                  "Please make sure to adjust the folder name in the config file or save the model in there.",
                  "Eventually, this can be resolved by running the train procedure with experiment number specified as -1.")
            print("Terminating the prediction process.")
            return

        showcase_model(gestures, examples=example_dir,
                       predict=True, model=model,
                       translations=config["Paths"]["Translations"],
                       img_size=config["General parameters"]["Image size"])


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
