# -*- coding: utf-8 -*-
"""
A simple function to enable running
the different scripts from command line.

Parameters:
    setup: string
        Setup the folder environment.
    collect: string
        Start the process of data collection.
    train: string
        Train the convolutional neural network.
    showcase: string
        Starts the camera and showcases the model prediction using pretrained model.

Default process (when no parameters given) is model showcasing

Inspired by Peter Mitura (https://github.com/PMitura/snli-rnn)
"""

import sys
import setup
import data_collection
import CNN
import model_showcase


def main(argv):

    if not len(argv):
        print("No arguments specified, will try model showcasing.")
        model_showcase.run()

    for arg in argv:
        if arg == "setup":
            setup.run()
        elif arg == "collect":
            data_collection.run()
        elif arg == "train":
            # TODO
            pass
        elif arg == "showcase":
            model_showcase.run()


if __name__ == "__main__":
    main(sys.argv[1:])
