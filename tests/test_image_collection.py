# -*- coding: utf-8 -*-
"""
Test file for image_collection module
"""

import os
import pytest
import image_collection
from tensorflow.keras.models import Sequential


# Testing invalid inputs for new_folder function
@pytest.mark.parametrize(
    "dir_name, verbose",
    [
        # Test invalid inputs for dir_name parameter
        (150, True),
        (0.5, True),
        (["STR"], True),
        (("ABCD",), True),

        # Test invalid inputs for verbose parameter
        ("Valid", 1),
        ("Valid", 0),
        ("Valid", "True"),
        ("Valid", "False"),
        ("Valid", []),
        ("Valid", "")
    ])
def test_new_folder_invalid(dir_name, verbose):

    with pytest.raises(ValueError):
        image_collection.new_folder(dir_name, verbose)


# Testing valid inputs for new_folder function
@pytest.mark.parametrize(
    "dir_name",
    [
        # Test valid inputs
        ("test1"),
        ("test/folder")
    ])
def test_new_folder_valid(tmpdir, dir_name):

    new_path = tmpdir.join(dir_name)
    assert not new_path.exists()
    image_collection.new_folder(str(new_path))

    if new_path == "test/folder":
        assert not new_path.exists()
    else:
        assert new_path.exists()


# Testing invalid inputs for repair_padding function
@pytest.mark.parametrize(
    "dir_name",
    [
        # Test invalid inputs for dir_name parameter
        (150),
        (0.5),
        (["STR"]),
        (("ABCD",))
    ])
def test_repair_padding_invalid(dir_name):

    with pytest.raises(ValueError):
        image_collection.repair_padding(dir_name)


# Testing repair_padding function
@pytest.mark.parametrize(
    "files, expected_files",
    [
        # Test various possibilities with and without gaps in the numbering
        (["A_1.jpg"], ["A_1.jpg"]),
        (["A_1.jpg", "A_2.jpg"], ["A_1.jpg", "A_2.jpg"]),
        (["A_1.jpg", "A_3.jpg", "A_4.jpg"], ["A_1.jpg", "A_2.jpg", "A_3.jpg"]),

        # Test with a single file with an invalid name
        (["A_1.jpg", "B_2.jpg"], ["A_1.jpg", "B_2.jpg"]),

        # Test repairing padding with a single file with a non-positive numbering
        (["A_0.jpg"], ["A_1.jpg"]),

        # Test repairing padding with two files, one with non-positive numbering and A_1 already existing
        (["A_1.jpg", "A_0.jpg"], ["A_1.jpg", "A_2.jpg"]),
    ])
def test_repair_padding(tmpdir, files, expected_files):

    for file in files:
        file_path = tmpdir.join(file)
        file_path.write("")

    image_collection.repair_padding(str(tmpdir))

    assert os.listdir(tmpdir) == expected_files


# Testing invalid inputs for setup_folders function
@pytest.mark.parametrize(
    "script_directory, gestures_list, amount_per_gesture",
    [
        # Test invalid inputs for script_directory parameter
        (123, ["A", "B"], 10),
        ("****/////", ["A", "B"], 10),
        ("invalid/directory/path", ["A", "B"], 10),

        # Test invalid inputs for gestures_list parameter
        ("C:/", 123, 10),
        ("C:/", [123, "B"], 10),
        ("C:/", ["A", "A"], 10),

        # Test invalid inputs for amount_per_gestures parameter
        ("C:/", ["gesture1", "gesture2"], "a"),
        ("C:/", ["gesture1", "gesture2"], -10)
    ])
def test_setup_folders_invalid(script_directory, gestures_list, amount_per_gesture):

    with pytest.raises(ValueError):
        image_collection.setup_folders(script_directory, gestures_list, amount_per_gesture)


# Testing valid inputs for setup_folders function
@pytest.mark.parametrize(
    "script_directory, gestures_list, amount_per_gesture",
    [
        (None, ["A", "B"], 10),
        ("test1", ["A", "B", "C"], 10),
        ("test2", ["A", "B", "C"], 100)
    ])
def test_setup_folders_valid(script_directory, gestures_list, amount_per_gesture, tmpdir):

    tmp_dir = str(tmpdir)

    # Create the test_dir directory if it doesn't already exist
    if script_directory is not None:
        test_dir = os.path.join(tmp_dir, script_directory)
        os.makedirs(test_dir)
    else:
        test_dir = tmp_dir

    data_dir, example_dir, desired_amount, current_amount, paths = image_collection.setup_folders(test_dir, gestures_list, amount_per_gesture)

    # Check that the returned values are as expected
    assert isinstance(data_dir, str)
    assert isinstance(example_dir, str)
    assert isinstance(desired_amount, dict)
    assert isinstance(current_amount, dict)
    assert isinstance(paths, dict)

    assert len(desired_amount) == len(gestures_list)
    assert all(isinstance(key, str) and isinstance(value, int) for key, value in desired_amount.items())
    assert all(isinstance(key, str) and isinstance(value, int) for key, value in current_amount.items())
    assert all(isinstance(key, str) and isinstance(value, str) for key, value in paths.items())

    # Check that the Data and Examples directories were created
    assert os.path.exists(data_dir)
    assert os.path.exists(example_dir)

    # Check that the subfolders for each gesture were created
    for gesture in gestures_list:
        gesture_path = os.path.join(data_dir, gesture)
        assert os.path.exists(gesture_path)
        assert paths[gesture] == gesture_path


# Testing invalid inputs for create_rectangle function
@pytest.mark.parametrize(
    "origin, size_X, size_Y",
    [
        # Test invalid inputs for origin parameter
        ((1, "STR"), 2, 3),
        ((1, 2, 3), 2, 3),
        ((1, -2), 2, 3),
        ((), 2, 3),

        # Test invalid inputs for size_X parameter
        ((1, 2), "STR", 3),
        ((1, 2), -2, 3),

        # Test invalid inputs for size_Y parameter
        ((1, 2), 2, "STR"),
        ((1, 2), 2, -3)
    ])
def test_create_rectangle_invalid_inputs(origin, size_X, size_Y):

    with pytest.raises(ValueError):
        image_collection.create_rectangle(origin, size_X, size_Y)


# Testing valid inputs for create_rectangle function
@pytest.mark.parametrize(
    "origin, size_X, size_Y, expected",
    [
        ((1, 2), 3, 4, [(1, 2), (4, 2), (1, 6), (4, 6)]),
        ((0, 0), 10, 20, [(0, 0), (10, 0), (0, 20), (10, 20)]),
        ((2, 3), 5, 5, [(2, 3), (7, 3), (2, 8), (7, 8)])
    ])
def test_create_rectangle_valid_inputs(origin, size_X, size_Y, expected):

    assert image_collection.create_rectangle(origin, size_X, size_Y) == expected


# Testing invalid inputs for image_capturing function
@pytest.mark.parametrize(
    "gesture_list, examples, current_amounts, desired_amounts",
    [
        # Test invalid inputs for gesture_list parameter
        ((1, "STR"), "Examples", {"A": 5, "B": 5}, {"A": 5, "B": 5}),
        ("STR", "Examples", {"A": 5, "B": 5}, {"A": 5, "B": 5}),
        (["STR", "STR"], "Examples", {"A": 5, "B": 5}, {"A": 5, "B": 5}),
        (["STR", 1.5], "Examples", {"A": 5, "B": 5}, {"A": 5, "B": 5}),

        # Test invalid inputs for examples parameter
        (["STR", "STR2"], 5, {"A": 5, "B": 5}, {"A": 5, "B": 5}),
        (["STR", "STR2"], 2.0, {"A": 5, "B": 5}, {"A": 5, "B": 5}),
        (["STR", "STR2"], "invalid/path", {"A": 5, "B": 5}, {"A": 5, "B": 5}),

        # Test invalid inputs for current_amounts parameter
        (["STR", "STR2"], "Examples", ["A", 5, "B", 10], {"A": 5, "B": 5}),
        (["STR", "STR2"], "Examples", {"A": 5, "B": -1}, {"A": 5, "B": 5}),
        (["STR", "STR2"], "Examples", {"A": 5, "B": 5.5}, {"A": 5, "B": 5}),
        (["STR", "STR2"], "Examples", {"A": 5, 10: 5}, {"A": 5, "B": 5}),

        # Test invalid inputs for desired_amounts parameter
        (["STR", "STR2"], "Examples", {"A": 5, "B": 5}, ["A", 5, "B", 10]),
        (["STR", "STR2"], "Examples", {"A": 5, "B": 5}, {"A": 5, "B": 5.5}),
        (["STR", "STR2"], "Examples", {"A": 5, "B": 5}, {"A": 5, 10: 5})
    ])
def test_image_capturing_invalid(gesture_list, examples, current_amounts, desired_amounts):

    with pytest.raises(ValueError):
        image_collection.image_capturing(gesture_list, examples,
                                         current_amounts=current_amounts, desired_amounts=desired_amounts)


# Another invalid inputs test for image_capturing function
@pytest.mark.parametrize(
    "predict, model",
    [
        # Test invalid inputs for predict parameter
        (0, Sequential()),
        (1, Sequential()),
        ("True", Sequential()),
        ([], Sequential()),

        # Test invalid inputs for model parameter
        (True, "Sequential()"),
        (True, 15),
        (True, None)
    ])
def test_image_capturing_invalid2(predict, model):

    with pytest.raises(ValueError):
        image_collection.image_capturing(gesture_list=["A", "B", "C"], predict=predict, model=model)
