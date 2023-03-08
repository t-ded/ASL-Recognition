# TODO: Build a function supporting building instructions in the form of ["i", "c", "c", "p", "d", "o"]
# TODO: If the instructions do not include "i" as first and "o" as last argument, add the input and output layers anyway
# TODO: The output layer should be a dense layer with the softmax activation
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
            adjusted based on this.
        instructions: str (default "I,O")
            Instructions for the architecture of the model.
            The layers should be split by ',' while the parameters
            for the respective layers should be split by '-'.
            Note that the input and output layers are added automatically
            if the instructions do not include them as the first and last layers.
                Example: "I,C-f64-w3-s2-pv,C-f64-w3-s2-pv,P-p2-s2-ps,D-0.5,H-100,O"
                    Creates a model with:
                        - input layer with shape inp_shape
                        - two convolutional layers with 64 filters,
                        window_size 3 and striding (2, 2) and padding valid
                        - pooling layer with pooling_size 2, striding (2, 2) and padding same
                        - dropout layer with dropout_rate 0.5
                        - densely connected layer with 100 units
                        - output layer of size output_size

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
        wrn = "The instructions did not include an input layer on the first position.\n"
        wrn += "The input layer will be added automatically.\n"
        warnings.warn(wrn)

    # Make sure the model ends with an output layer and inform the user
    if instructions[-1] != "O":
        wrn = "The instructions did not include an output layer on the last position.\n"
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
            wrn = "One of the layers in the instructions given was empty,\n"
            wrn += "i.e. the instructions parameter contains ',,'.\n"
            wrn += "Omitting the layer and continuing the process.\n"
            warnings.warn(wrn)
            continue

        # Set up the name of the layer
        layer_name = layer[0]

        if len(layer) == 1 and layer_name != "O":
            wrn = "One of the hidden layers does not have specified parameters.\n"
            wrn += "Omitting the layer and continuing the process.\n"
            warnings.warn(wrn)
            continue

        # Continue the process based on the type of the layer

        # Convolutional layer
        if layer_name == "C":
            pass

        # Pooling layer
        if layer_name == "P":
            pass

        # Dropout layer
        if layer_name == "D":

            # Extract configuration
            pattern = r"H-(\d*)"
            match = re.match(pattern, layer_name)

            # Ensure correct input
            if not match:
                wrn = "The argument for the dropout layer is not specified.\n"
                wrn += "Omitting the layer and continuing the process.\n"
                warnings.warn(wrn)
                continue

            rate = float(match.group(1))
            if not 0 <= rate <= 1:
                wrn = "The argument for the dropout layer is not a number between 0 and 1.\n"
                wrn += "Omitting the layer and continuing the process.\n"
                warnings.warn(wrn)
                continue

            hidden = tf.keras.layers.Dropout(rate)(hidden)

        # Densely connected layer
        if layer_name == "H":

            # Extract configuration
            pattern = r"H-(\d*)"
            match = re.match(pattern, layer_name)

            # Ensure correct input
            if not match:
                wrn = "The argument for the dense layer is not specified.\n"
                wrn += "Omitting the layer and continuing the process.\n"
                warnings.warn(wrn)
                continue

            hidden = tf.keras.layers.Dense(int(match.group(1)))(hidden)

        # Output layer
        if layer_name == "O":

            # Adjust the output layer activation based on the output_size
            if output_size == 1:
                output = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)(hidden)
            else:
                output = tf.keras.layers.Dense(output_size, activation=tf.nn.relu)(hidden)

        # Invalid layer name
        else:
            wrn = f"Invalid character was given as layer name: {layer_name}.\n"
            wrn += "Omitting the layer and continuing the process.\n"
            continue

    return tf.keras.Model(inputs=inp, outputs=output)
