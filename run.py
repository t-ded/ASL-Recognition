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

import sys
import os
import collect_dataset
import CNN
import utils
import tensorflow as tf


gestures = ["I index", "My", "You", "Your",
            "In", "To", "With", "Yes",
            "No", "Well", "I love you",
            "Oh I see", "Name", "Hug",
            "Internet", "Bus", "Money",
            "Work", "Ask", "Go",
            "Look", "Have", "Correct",
            "Want", "Where",
            "A", "B", "C", "D",
            "E", "F", "G", "H",
            "I", "K", "L", "M",
            "N", "O", "P", "Q",
            "R", "S", "T", "U",
            "V", "W", "X", "Y"]


def main(argv):
    """Command line function"""

    if not argv:
        print("No arguments specified, will try to run the whole procedure.")
        argv = ["collect", "train", "showcase"]

    for arg in argv:
        if arg == "collect":
            data_dir, example_dir, desired_amount, current_amount, paths = utils.setup_folders(os.path.dirname("image_collection.py"), gestures, 10)
            print("The folders have been set up.")
            print("Starting the data collection process.")
            collect_dataset.collect_data(gestures, examples=example_dir,
                                             data_directory=data_dir, current_amounts=current_amount,
                                             desired_amounts=desired_amount, gesture_paths=paths, translations="translations.txt", img_size=196)
            print("Your data has been collected, please check the folders.")
        elif arg == "train":
            data_dir, example_dir, desired_amount, current_amount, paths = utils.setup_folders(os.path.dirname("image_collection.py"), gestures, 10)
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
            model.save_weights("Weights/weights")
        elif arg == "showcase":
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
            image_collection.image_capturing(gestures, examples=example_dir, save=False, predict=True, data_directory=data_dir, model=model)
        else:
            print(f"Unknown argument {arg}")


if __name__ == "__main__":
    main(sys.argv[1:])
