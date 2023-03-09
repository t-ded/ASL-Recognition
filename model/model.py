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
from preprocessing import Grayscale, AdaptiveThresholding, Blurring


def build_model(inp_shape, output_size, name="model", instructions="I,O"):
    """
    Build model with the specified architecture

    Parameters:
        inp_shape: list of ints
            Dimensions of the input (excluding the size of the batch).
        output_size: int
            Dimensions of the output. Output layer activation is automatically
            adjusted based on this (sigmoid for 1 and softmax otherwise)
        name: str (default "model")
            A name for the output model.
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
                        kernel_size 3 and striding (2, 2)
                        - max pooling layer with pooling_size 2, striding (2, 2)
                        - dropout layer with dropout_rate 0.5
                        - flatten layer
                        - densely connected layer with 100 units
                        - output layer of size output_size (softmax activation for output_size > 1)

                Supported layers and their supported arguments:
                    - Input: I
                    - Convolutional (filters(f),
                                     kernel_size(k),
                                     strides(s)): C-f10-k3-s2
                    - Pooling (pool_size(p),
                               strides(s),
                               type(t, one of average (a) or max (m))): P-p2-s2-ta
                    - Flatten: F
                    - Dropout (rate): D-0.5
                    - Dense (units): H-100
                    - Output: O

    Returns:
        model: tf.keras.Model
            Instance of an uncompiled keras model (functional).
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

    if not isinstance(name, str):
        raise ValueError("Different datatype than string has been given as input for the parameter name")

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
    inp = tf.keras.layers.Input(shape=inp_shape, name="trainable_input")
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

        if len(layer) == 1 and layer_name not in ["I", "O", "F"]:
            wrn = "\nOne of the hidden layers does not have specified parameters.\n"
            wrn += "Omitting the layer and continuing the process.\n"
            warnings.warn(wrn)
            continue

        # Continue the process based on the type of the layer

        # Convolutional layer
        if layer_name == "C":

            # Ensure the number of filters is specified
            filters = re.search(r"-f(\d*)", layer).group(1)
            if not filters:
                wrn = "\nThe number of filters for the convolutional layer is not specified.\n"
                wrn += "Omitting the layer and continuing the process.\n"
                warnings.warn(wrn)
                continue

            # Ensure the kernel size is specified
            kernel_size = re.search(r"-k(\d*)", layer).group(1)
            if not kernel_size:
                wrn = "\nThe kernel_size for the convolutional layer is not specified.\n"
                wrn += "Omitting the layer and continuing the process.\n"
                warnings.warn(wrn)
                continue

            # If strides is not specified, set it to default
            strides = re.search(r"-k(\d*)", layer).group(1)
            if not strides:
                strides = 1

            hidden = tf.keras.layers.Conv2D(filters=int(filters),
                                            kernel_size=int(kernel_size),
                                            strides=int(strides))(hidden)

        # Pooling layer
        elif layer_name == "P":

            # Ensure the type of the pooling layer is specified
            pooling_type = re.search(r"-t(\w)", layer).group(1)
            if not pooling_type:
                wrn = "\nThe type for the pooling layer is not specified.\n"
                wrn += "Omitting the layer and continuing the process.\n"
                warnings.warn(wrn)
                continue

            # Ensure the pooling size is specified
            pool_size = re.search(r"-p(\d*)", layer).group(1)
            if not pool_size:
                wrn = "\nThe pool_size for the pooling layer is not specified.\n"
                wrn += "Omitting the layer and continuing the process.\n"
                warnings.warn(wrn)
                continue

            # If strides is not specified, set it to default
            strides = re.search(r"-s(\d*)", layer).group(1)
            if not strides:
                strides = None
            else:
                strides = int(strides)

            # Choose the correct type of the pooling layer
            if pooling_type == "a":
                hidden = tf.keras.layers.AveragePooling2D(pool_size=int(pool_size),
                                                          strides=strides)(hidden)
            elif pooling_type == "m":
                hidden = tf.keras.layers.MaxPool2D(pool_size=int(pool_size),
                                                   strides=strides)(hidden)
            else:
                wrn = "\nThe type for the pooling layer is not valid.\n"
                wrn += "Omitting the layer and continuing the process.\n"
                warnings.warn(wrn)
                continue

        # Dropout layer
        elif layer_name == "D":

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
        elif layer_name == "H":

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
        elif layer_name == "F":

            hidden = tf.keras.layers.Flatten()(hidden)

        # Output layer
        elif layer_name == "O":

            # Adjust the output layer activation based on the output_size
            if output_size == 1:
                output = tf.keras.layers.Dense(1,
                                               activation=tf.nn.sigmoid,
                                               name="sigmoid_output")(hidden)
            else:
                output = tf.keras.layers.Dense(output_size,
                                               activation=tf.nn.softmax,
                                               name="softmax_output")(hidden)

        # Invalid layer name
        else:
            wrn = f"Invalid character was given as layer name: {layer_name}.\n"
            wrn += "Omitting the layer and continuing the process.\n"
            continue

    return tf.keras.Model(inputs=inp, outputs=output, name=name)


def build_preprocessing(inp_shape, name="preprocessing", instructions="I,G"):
    """
    Build preprocessing pipeline model with the specified architecture

    Parameters:
        inp_shape: list of ints
            Dimensions of the input (excluding the size of the batch).
        name: str (default "preprocessing")
            A name for the output model.
        instructions: str (default "I")
            Instructions for the architecture of the preprocessing pipeline.
            The layers should be split by ',' while the parameters
            for the respective layers should be split by '-'.
            Note that the input layer is added automatically
            if the instructions do not include it as the first layer.
            The option to select output shape is omitted, note that using
            a grayscale layer changes the number of channels to 1. Among other
            things, this means that the instructions cannot contain more
            than one grayscale layer.
            Adaptive thresholding can only be used post-grayscale layer.

                Example: "I,G,B-tg-k3-s1,T-tm-b3-c(-3),R"
                    Creates a model with:
                        - input layer with shape inp_shape
                        - grayscale layer
                        - blurring layer with type gaussian, kernel_size 3 and sigma 1
                        - adaptive thresholding layer with type mean, block_size 3 and constant -3
                        - rescaling layer

                Supported layers and their supported arguments:
                    - Input: I
                    - Grayscale: G
                    - Blurring (type(t, one of gaussian(g) or median(m)),
                                kernel_size(k),
                                sigma(s, only relevant for gaussian blurring)): B-tg-k3-s2
                    - AdaptiveThresholding (type(t, one of gaussian(g) or mean(m)),
                                            block_size(b),
                                            constant(c, specified in parentheses)): T-tm-b3-c(-3)
                    - Rescale: R

    Returns:
        model: tf.keras.Model
            Instance of an uncompiled keras model (functional) without trainable layers.
    """
    # Input management
    if not isinstance(inp_shape, list):
        raise ValueError("Different datatype than list has been given as input for the parameter inp_shape.")
    for val in inp_shape:
        if not isinstance(val, int):
            raise ValueError("Elements of the inp_shape list are not integers.")
        if not val > 0:
            raise ValueError("The dimensions of the input must be positive.")

    if not isinstance(name, str):
        raise ValueError("Different datatype than string has been given as input for the parameter name")

    # Further input management for the instructions variable is performed
    # per layer during the pipeline building process
    if not isinstance(instructions, str):
        raise ValueError("A string was expected as input for the instructions parameter")

    # Check if not more than 1 grayscale layer is present in the instructions
    grayscale_limit = 1
    if instructions.count("G") > 1:
        wrn = "\nThe instructions for the preprocessing pipeline contained more than one grayscale layer.\n"
        wrn += "Such pipeline is not valid, thus the latter grayscale layer(s) will be omitted.\n"
        warnings.warn(wrn)

    # Inform the user the model will start with an input layer
    if instructions[0] != "I":
        wrn = "\nThe instructions did not include an input layer on the first position.\n"
        wrn += "The input layer will be added automatically.\n"
        warnings.warn(wrn)

    # Initialize the input layer
    inp = tf.keras.layers.Input(shape=inp_shape, name="preprocessing_input")
    preprocessing = inp

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

        if len(layer) == 1 and layer_name in ["B", "T"]:
            wrn = "\nOne of the blurring or thresholding layers does not have specified parameters.\n"
            wrn += "Omitting the layer and continuing the process.\n"
            warnings.warn(wrn)
            continue

        # Continue the process based on the type of the layer

        # Grayscale layer
        if layer_name == "G":

            # Ensure the process only continues for the first grayscale layer
            if grayscale_limit:
                preprocessing = Grayscale()(preprocessing)
                grayscale_limit = 0

            continue

        # Blurring layer
        if layer_name == "B":

            # Ensure the type of the blurring layer is specified
            blurring_type = re.search(r"-t(\w)", layer).group(1)
            if not blurring_type:
                wrn = "\nThe type for the blurring layer is not specified.\n"
                wrn += "Omitting the layer and continuing the process.\n"
                warnings.warn(wrn)
                continue

            # Choose the correct type of the blurring layer
            if blurring_type == "g":
                blurring_type = "gaussian"

            elif blurring_type == "m":
                blurring_type = "median"

            else:
                wrn = "\nThe type for the blurring layer is not valid.\n"
                wrn += "Omitting the layer and continuing the process.\n"
                warnings.warn(wrn)
                continue

            # If kernel size is not specified, set it to default
            kernel_size = re.search(r"-k(\d*)", layer).group(1)
            if not kernel_size:
                kernel_size = 3
                wrn = "\nThe kernel_size for the blurring layer is not specified.\n"
                wrn += f"Setting the kernel_size to default ({kernel_size}) and continuing the process.\n"
                warnings.warn(wrn)

            # If sigma is not specified, set it to default (only relevant for gaussian blur)
            sigma = re.search(r"s([\d\.]*)", layer).group(1)
            if not sigma:
                sigma = 1
                if blurring_type == "gaussian":
                    wrn = "\nThe sigma parameter for the gaussian blurring layer is not specified.\n"
                    wrn += f"Setting the sigma parameter to default ({sigma}) and continuing the process.\n"
                    warnings.warn(wrn)

            preprocessing = Blurring(blurring_type=blurring_type,
                                     kernel_size=int(kernel_size),
                                     sigma=float(sigma))(preprocessing)

        # Adaptive thresholding layer
        if layer_name == "T":

            # Ensure the settings for the thresholding layer are valid (i.e. preceded by a grayscale layer)
            if grayscale_limit:
                wrn = "\nThe instructions for the preprocessing pipeline contains\n"
                wrn += "an adaptive thresholding layer that was not preceded by a grayscale layer.\n"
                wrn += "Such pipeline is not valid, thus this adaptive thresholding layer will be omitted.\n"
                warnings.warn(wrn)
                continue

            # Ensure the type of the thresholding layer is specified
            thresholding_type = re.search(r"-t(\w)", layer).group(1)
            if not thresholding_type:
                wrn = "\nThe type for the thresholding layer is not specified.\n"
                wrn += "Omitting the layer and continuing the process.\n"
                warnings.warn(wrn)
                continue

            # Choose the correct type of the thresholding layer
            if thresholding_type == "g":
                thresholding_type = "gaussian"

            elif thresholding_type == "m":
                thresholding_type = "mean"

            else:
                wrn = "\nThe type for the thresholding layer is not valid.\n"
                wrn += "Omitting the layer and continuing the process.\n"
                warnings.warn(wrn)
                continue

            # If block size is not specified, set it to default
            block_size = re.search(r"-b(\d*)", layer).group(1)
            if not block_size:
                block_size = 3
                wrn = "\nThe kernel_size for the blurring layer is not specified.\n"
                wrn += f"Setting the kernel_size to default ({kernel_size}) and continuing the process.\n"
                warnings.warn(wrn)

            # If constant is not specified, set it to default
            constant = re.search(r"c([\d\.]*)", layer).group(1)
            if not constant:
                constant = 0
                wrn = "\nThe constant for the adaptive thresholding layer is not specified.\n"
                wrn += f"Setting the constant to default ({constant}) and continuing the process.\n"
                warnings.warn(wrn)

            preprocessing = AdaptiveThresholding(thresholding_type=thresholding_type,
                                                 block_size=int(block_size),
                                                 constant=float(constant))(preprocessing)

        # Rescaling layer
        elif layer_name == "G":

            preprocessing = tf.keras.layers.Rescaling(scale=(1. / 255))(preprocessing)

    return tf.keras.Model(inputs=inp, outputs=preprocessing, name=name)
