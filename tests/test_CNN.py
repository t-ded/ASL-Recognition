# -*- coding: utf-8 -*-
"""
Test file for CNN module
"""

import pytest
import CNN


# Testing invalid inputs for build_model function
@pytest.mark.parametrize(
    "labels, convolutional_layers, pooling_layers, input_shape",
    [
        # Test invalid inputs for labels parameter
        (-1, 3, True, (10, 10)),
        (0.5, 3, True, (10, 10)),
        ("STR", 3, True, (10, 10)),
        (False, 3, True, (10, 10)),
        ([], 3, True, (10, 10)),

        # Test invalid inputs for convolutional_layers parameter
        (10, -1, True, (10, 10)),
        (10, 0.5, True, (10, 10)),
        (10, "STR", True, (10, 10)),
        (10, [], True, (10, 10)),

        # Test invalid inputs for pooling_layers parameter
        (10, 3, -1, (10, 10)),
        (10, 3, 0.5, (10, 10)),
        (10, 3, "STR", (10, 10)),
        (10, 3, ("A", "B"), (10, 10)),
        (10, 3, [], (10, 10)),

        # Test invalid inputs for input_shape parameter
        (10, 3, True, (5, 7, -1)),
        (10, 3, True, (0.6, 12)),
        (10, 3, True, "STR"),
        (10, 3, True, [10, 10])
    ])
def test_build_model_invalid(labels, convolutional_layers, pooling_layers, input_shape):

    with pytest.raises(ValueError):
        CNN.build_model(labels, convolutional_layers, pooling_layers, input_shape)


# Testing valid inputs for build_model function
@pytest.mark.parametrize(
    "labels, convolutional_layers, pooling_layers, input_shape",
    [
        # Test valid inputs
        (3, 3, True, (45, 45)),
        (7, 2, False, (35, 35, 35)),
        (3, 1, True, (65, 35)),
        (3, 5, False, (35, 85, 35))
    ])
def test_build_model_valid(labels, convolutional_layers, pooling_layers, input_shape):

    model = CNN.build_model(labels, convolutional_layers, pooling_layers, input_shape)

    # Input & output + fully-connected + convolutional layers + pooling layers if present
    expected_num_layers = 2 + 1 + convolutional_layers + int(pooling_layers) * (convolutional_layers - 1)
    assert len(model.layers) == expected_num_layers
    assert model.layers[-1].units == labels
    assert model.layers[0].input_shape[1:(1 + len(input_shape))] == input_shape
