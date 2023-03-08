"""
Module to enable tf.keras.Model object building based on given text instructions
for architecture settings. The module is used for the the Bachelor's Thesis project
on the following topic:
    "Construction of a Neural Networks model for translation of recorded sign language".

@author: Tomáš Děd
"""

import warnings
import re
import tensorflow as tf


def build_model(inp_shape, output_size, instructions="I,O"):
    """
    Build model with the specified architecture

    Parameters:
        inp_shape: list of ints
            Dimensions of the input (excluding the size of the batch).
        output_size: int
            Dimensions of the output. Output layer activation is automatically
            adjusted based on this (sigmoid for 1 and softmax otherwise)
        instructions: str (default "I,O")
            Instructions for the architecture of the model.
            The layers should be split by ',' while the parameters
            for the respective layers should be split by '-'.
            Note that the input and output layers are added automatically
            if the instructions do not include them as the first and last layers.

                Example: "I,C-f64-k3-s2,C-f64-k3-s2,P-ps2-s2-tm,D-0.5,F,H-100,O"
                    Creates a model with:
                        - input layer with shape inp_shape
                        - two convolutional layers with 64 filters,
                        window_size 3 and striding (2, 2)
                        - max pooling layer with pooling_size 2, striding (2, 2)
                        - dropout layer with dropout_rate 0.5
                        - flatten layer
                        - densely connected layer with 100 units
                        - output layer of size output_size (softmax activation for output_size > 1)

                Supported layers and their supported arguments:
                    - Input: I
                    - Convolutional (filters, kernel_size, strides): C-f10-k3-s2
                    - Pooling (pool_size, strides, type(one of average (a) or max (m))): P-p2-s2-ta
                    - Flatten: F
                    - Dropout (rate): D-0.5
                    - Dense (units): H-100
                    - Output: O

    Returns:
        model: tf.keras.Model
            Instance of an uncompiled keras model.
    """
    # Input management
    if not isinstance(inp_shape, list):
        raise ValueError("Different datatype than list has been given as input for the parameter inp_shape.")
    for val in inp_shape:
        if not isinstance(val, int):
            raise ValueError("Elements of the inp_shape list are not integers.")
        if not val > 0:
            raise ValueError("The dimensions of the input must be positive.")

    if not isinstance(output_size, int):
        raise ValueError("Different datatype than integer has been given as input for the parameter output_size.")

    if output_size <= 0:
        raise ValueError("The dimensions of the output must be positive.")

    # Further input management for the instructions variable is performed
    # per layer during the model building process
    if not isinstance(instructions, str):
        raise ValueError("A string was expected as input for the instructions parameter")

    # Inform the user the model will start with an input layer
    if instructions[0] != "I":
        wrn = "\nThe instructions did not include an input layer on the first position.\n"
        wrn += "The input layer will be added automatically.\n"
        warnings.warn(wrn)

    # Make sure the model ends with an output layer and inform the user
    if instructions[-1] != "O":
        wrn = "\nThe instructions did not include an output layer on the last position.\n"
        wrn += "The output layer will be added automatically.\n"
        warnings.warn(wrn)
        instructions += ",O"

        # Make sure there is no output layer inside the model
        if instructions.index("O") != len(instructions) - 1:
            raise ValueError("The instructions for the model included output layer somewhere else than as the last layer.")

    # Initialize the input layer
    inp = tf.keras.layers.Input(shape=inp_shape)
    hidden = inp

    # Parse the instructions
    layers = instructions.split(",")

    # Chain the layers based on the instructrions
    for layer in layers:

        # Empty instruction
        if not layer:
            wrn = "\nOne of the layers in the instructions given was empty,\n"
            wrn += "i.e. the instructions parameter contains ',,'.\n"
            wrn += "Omitting the layer and continuing the process.\n"
            warnings.warn(wrn)
            continue

        # Set up the name of the layer
        layer_name = layer[0]

        if len(layer) == 1 and layer_name != "O":
            wrn = "\nOne of the hidden layers does not have specified parameters.\n"
            wrn += "Omitting the layer and continuing the process.\n"
            warnings.warn(wrn)
            continue

        # Continue the process based on the type of the layer

        # Convolutional layer
        if layer_name == "C":

            # Extract configuration
            pattern = r"-f(\d*)-k(\d*)-s(\d*)"
            match = re.search(pattern, layer)

            # Ensure correct input
            if not match:
                wrn = "\nThe argument for the convolutional layer is not specified.\n"
                wrn += "Omitting the layer and continuing the process.\n"
                warnings.warn(wrn)
                continue

            # Ensure the number of filters is specified
            filters = match.group(1)
            if not filters:
                wrn = "\nThe number of filters for the convolutional layer is not specified.\n"
                wrn += "Omitting the layer and continuing the process.\n"
                warnings.warn(wrn)
                continue

            # Ensure the kernel size is specified
            kernel_size = match.group(2)
            if not filters:
                wrn = "\nThe kernel_size for the convolutional layer is not specified.\n"
                wrn += "Omitting the layer and continuing the process.\n"
                warnings.warn(wrn)
                continue

            # If strides is not specified, set it to default
            strides = match.group(3)
            if not strides:
                strides = 1

            hidden = tf.keras.layers.Conv2D(filters=int(filters),
                                            kernel_size=int(kernel_size),
                                            strides=int(strides))(hidden)

        # Pooling layer
        if layer_name == "P":

            # Extract configuration
            pattern = r"-t(\w)-p(\d*)-s(\d*)"
            match = re.search(pattern, layer)

            # Ensure correct input
            if not match:
                wrn = "\nThe argument for the pooling layer is not specified.\n"
                wrn += "Omitting the layer and continuing the process.\n"
                warnings.warn(wrn)
                continue

            # Ensure the type of the pooling layer is specified
            pooling_type = match.group(1)
            if not pooling_type:
                wrn = "\nThe type for the pooling layer is not specified.\n"
                wrn += "Omitting the layer and continuing the process.\n"
                warnings.warn(wrn)
                continue

            # Ensure the pooling size is specified
            pool_size = match.group(2)
            if not filters:
                wrn = "\nThe pool_size for the pooling layer is not specified.\n"
                wrn += "Omitting the layer and continuing the process.\n"
                warnings.warn(wrn)
                continue

            # If strides is not specified, set it to default
            strides = match.group(3)
            if not strides:
                strides = None
            else:
                strides = int(strides)

            # Choose the correct type of the pooling layer
            if pooling_type == "a":
                hidden = tf.keras.layers.AveragePooling2D(pool_size=pool_size,
                                                          strides=strides)(hidden)
            elif pooling_type == "m":
                hidden = tf.keras.layers.MaxPool2D(pool_size=pool_size,
                                                   strides=strides)(hidden)
            else:
                wrn = "\nThe type for the pooling layer is not valid.\n"
                wrn += "Omitting the layer and continuing the process.\n"
                warnings.warn(wrn)
                continue

        # Dropout layer
        if layer_name == "D":

            # Extract configuration
            pattern = r"D-([\d\.]*)"
            match = re.search(pattern, layer)

            # Ensure the dropout rate is specified
            if not match:
                wrn = "\nThe argument for the dropout layer is not specified.\n"
                wrn += "Omitting the layer and continuing the process.\n"
                warnings.warn(wrn)
                continue

            # Ensure correct input
            rate = float(match.group(1))
            if not 0 <= rate <= 1:
                wrn = "\nThe argument for the dropout layer is not a number between 0 and 1.\n"
                wrn += "Omitting the layer and continuing the process.\n"
                warnings.warn(wrn)
                continue

            hidden = tf.keras.layers.Dropout(rate)(hidden)

        # Densely connected layer
        if layer_name == "H":

            # Extract configuration
            pattern = r"H-(\d*)"
            match = re.search(pattern, layer)

            # Ensure the number of units is specified
            if not match:
                wrn = "\nThe argument for the dense layer is not specified.\n"
                wrn += "Omitting the layer and continuing the process.\n"
                warnings.warn(wrn)
                continue

            hidden = tf.keras.layers.Dense(int(match.group(1)))(hidden)

        # Flatten layer
        if layer_name == "F":

            hidden = tf.keras.layers.Flatten()(hidden)

        # Output layer
        if layer_name == "O":

            # Adjust the output layer activation based on the output_size
            if output_size == 1:
                output = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)(hidden)
            else:
                output = tf.keras.layers.Dense(output_size, activation=tf.nn.softmax)(hidden)

        # Invalid layer name
        else:
            wrn = f"Invalid character was given as layer name: {layer_name}.\n"
            wrn += "Omitting the layer and continuing the process.\n"
            continue

    return tf.keras.Model(inputs=inp, outputs=output)
