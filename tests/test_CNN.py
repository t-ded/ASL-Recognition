# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 17:26:31 2022

@author: tomas
"""

import cv2
import pytest
import numpy as np
import CNN


# Testing invalid inputs
@pytest.mark.parametrize(
    "labels, convolutional_layers, pooling_layers, input_shape",
    [
        (-1, 0, "STR", 1, []),
        ((5, 7, -1), ["A"], "STR", 0),

        (-1, 3, True, (10, 10)),
        (0, 3, True, (10, 10)),
        ("STR", 3, True, (10, 10)),
        (True, 3, True, (10, 10)),
        ([], 3, True, (10, 10)),

        (10, -1, True, (10, 10)),
        (10, 0, True, (10, 10)),
        (10, "STR", True, (10, 10)),
        (10, True, True, (10, 10)),
        (10, [], True, (10, 10)),
    ])
def test_build_model_invalid(labels, convolutional_layers, pooling_layers, input_shape):

    assert CNN.build_model(labels, convolutional_layers, pooling_layers, input_shape) == ValueError


    
    
def build_model(labels, convolutional_layers=2, pooling_layers=True, input_shape=None):
    """
    Creates a simple convolutional network by adding layers based on input

    Parameters:
        labels: int
            Number of layers for prediction (i.e. size of output layer of the model). Should be positive and higher than 1.
        convolutional_layers: int (default 2)
            Number of convolutional layers (using tensorflows Conv2D, including the input convolutional layer). Should be positive.
        pooling_layers: bool (default True)
            Whether to include pooling layers between every pair of convolutional layers
        input_shape: (int, int, int) or (int, int) or None (default None):
            Shape of the image that will be given as input (i.e. input shape of the first layer).
            Integers in the tuple are required to be positive.

    Returns:
        model: keras.engine.sequential.Sequential
            Model created according to input
    """
    # Input management
    if not isinstance(labels, int):
        raise ValueError("Different datatype than integer has been given as input for the number of labels")

    if not isinstance(convolutional_layers, int):
        raise ValueError("Different datatype than integer has been given as input for the number of convolutional layers")

    if labels < 1:
        raise ValueError("Number of labels is less than 1. Please specify different amount.")

    if labels == 1:
        wrn = "\nYou have entered 1 as the number of labels.\n"
        wrn += "This might result in unpredicted behaviour and there is not much point in building a model then"
        warnings.warn(wrn)

    if convolutional_layers < 1:
        raise ValueError("This function expects at least one convolutional layer to be present in the model.")

    if not isinstance(pooling_layers, bool):
        raise ValueError("Different datatype than boolean has been given as input for the pooling_layers parameter")

    if input_shape:
        if not isinstance(input_shape, tuple):
            raise ValueError("Input shape has been assigned and different input than tuple was given")
        if len(input_shape) not in [2, 3]:
            raise ValueError("2D or 3D images expected as input")
        if len(input_shape) == 2:
            input_shape = tuple([*input_shape, 1])
        for val in input_shape:
            if not isinstance(val, int):
                raise ValueError("Integers were expected in place of image dimensions in parameter input_shape")
            if val < 0:
                raise ValueError("One of the dimensions of the input shape given is negative. Please give correct input shape.")

    model = models.Sequential()

    model.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=input_shape))
    for _ in range(convolutional_layers - 1):
        if pooling_layers:
            model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation="relu", input_shape=input_shape))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dense(labels))

    return model