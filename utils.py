"""
Multipurpose utility functions for the Bachelor's Thesis project
on the following topic:
    "Construction of a Neural Networks model for translation of recorded sign language".

@author: Tomáš Děd
"""

import os
import re
import warnings
import cv2
import numpy as np
from tensorflow.keras import layers
from keras_cv.layers import Grayscale
from model.preprocessing import AdaptiveThresholding, Blurring


def new_folder(dir_name, verbose=False):
    """
    Check whether given directory exists and create new one if not.

    Parameters:
        dir_name: string
            Name of the directory to be checked and to be created.
        verbose: bool (default False)
            Whether or not to inform the user about the creation of the folder.
    """
    # Input management
    if not isinstance(dir_name, str):
        raise ValueError("Different datatype than string has been given as input for name of the directory.")

    if not isinstance(verbose, bool):
        raise ValueError("Different datatype than boolean has been given as input for the verbose parameter")

    # Check directory's existence and try to create it if not existing and if possible
    if not os.path.exists(dir_name):
        try:
            os.makedirs(dir_name)
            if verbose:
                print(f"A folder has been successfully created with this path: {dir_name}")
        except OSError:
            wrn = "\nThe path you have specified for the new folder to-be created is invalid."
            wrn += "\nThe folder does not exist and was not created. Please specify a correct path."
            warnings.warn(wrn)

    elif verbose:
        print(f"A folder with this name already exists: {dir_name}")


def repair_padding(dir_name):
    """
    Images padded as A_1.jpg, A_2.jpg, ... are expected for the project.
    This function detects skips in padding (e.g. A_1.jpg, A_3.jpg) and repairs these.

    Parameters:
        dir_name: str
            Path to the directory that needs repair.
    """
    # Input management
    if not isinstance(dir_name, str):
        raise ValueError("Different datatype than string has been given as input for name of the directory.")

    if not os.path.exists(dir_name):
        warnings.warn("\nDirectory with given path does not exist. No padding adjustments have been made, returning.")
        return

    # Obtain the list of files in the folder
    files = os.listdir(dir_name)
    files.sort(key=lambda file: int(re.split(r"[_|.]", file)[1]))
    filecount = len(files)

    # The first image should have 1 in it's padding, rename it if necessary
    if filecount:
        first_split = re.split(r"[_|.]", files[0])
        last_split = re.split(r"[_|.]", files[-1])
        if int(first_split[1]) != 1:
            try:
                new_name = first_split[0] + "_" + str(1) + "." + first_split[2]
                os.rename(os.path.join(dir_name, files[0]), os.path.join(dir_name, new_name))
            except FileExistsError:
                new_name = first_split[0] + "_" + str(int(last_split[1]) + 1) + "." + first_split[2]
                os.rename(os.path.join(dir_name, files[0]), os.path.join(dir_name, new_name))
            files = os.listdir(dir_name)
            files.sort(key=lambda file: int(re.split(r"[_|.]", file)[1]))

    # Go through each file and if the run order skips a count, shift all the following files' padding
    for i in range(filecount - 1):
        name_split = re.split(r"[_|.]", files[i])
        name_split_next = re.split(r"[_|.]", files[i + 1])

        # Stop the process if there is some unexpected (i.e. < 1) number in padding
        if int(name_split_next[1]) < 1:
            wrn = f"\nIn the folder {dir_name} there is a file {files[i + 1]} with non-positive numbering!\n"
            wrn += "This might result in unexpected behaviour. The repair process will now terminate, please correct the number."
            warnings.warn(wrn)
            return

        # Warn the user if there is some mix of gestures
        if name_split[0] != name_split_next[0]:
            warnings.warn(f"There are files with different naming in the current directory ({dir_name}): {files[i]}, {files[i + 1]}!")

        # Find the padding jump
        diff = int(name_split_next[1]) - int(name_split[1])

        # The expected padding difference is one,
        # otherwise something is missing and all following files need to be shifted
        if diff != 1:
            for j in range(i + 1, filecount):
                name_current = re.split(r"[_|.]", files[j])
                new_name = name_current[0] + "_" + str(int(name_current[1]) - diff + 1) + "." + name_current[2]
                os.rename(os.path.join(dir_name, files[j]), os.path.join(dir_name, new_name))

            # Get the new list of files with adjusted padding
            files = os.listdir(dir_name)
            files.sort(key=lambda file: int(re.split(r"[_|.]", file)[1]))


def setup_folders(script_directory, gestures_list, amount_per_gesture):
    """
    Setup folders for the project. Add directories for data and examples
    and subfolders for each gesture in the list of gestures. Also setup
    the current and desired amounts for each gesture.

    Parameters:
        script_directory: str
            Name of the directory where the script is located. All folders
            will be created in this directory.
        gestures_list: list of strings
            List of all gestures to be included in the project.
            Subfolder will be created for each one in the Data folder.
            Example should be present for each one in Examples folder.
        amount_per_gesture: int
            Positive integer that specifies the desired number of datapoints per gesture.

    Returns:
        data_directory: str
            Directory name for the Data folder. Subfolders for each gesture are located in this folder.
        example_directory: str
            Directory name of the Examples folder.
            If not present beforehand, it is created and filled with dummy images per gesture.
        desired_amount: dict of str: int pairs
            Dictionary of desired amounts of samples per each gesture.
        current_amount: dict of str: int pairs
            Dictionary of current amounts of samples per each gesture.
        paths: dict of str: str
            Dictionary of paths to all the gestures.
    """
    # Input management
    if not isinstance(script_directory, str):
        raise ValueError("Different datatype than string has been given as input for name of the script directory.")
    if not script_directory:
        warnings.warn("Given script directory is empty, thus all the processes will run in the location of this script.")
    else:
        if not os.path.exists(script_directory):
            raise ValueError("The given directory for the script does not exist, please specify a correct directory path.")

    if not isinstance(gestures_list, list):
        raise ValueError("Different datatype than list has been given as input for the list of gestures.")
    for val in gestures_list:
        if not isinstance(val, str):
            raise ValueError("There is a value with different datatype than string in the list of gestures.")
    if len(gestures_list) != len(set(gestures_list)):
        raise ValueError("There are some gesture duplicates in the list of gestures, please check.")

    if not isinstance(amount_per_gesture, int):
        raise ValueError("Different datatype than integer has been given as input for the desired amount per gesture.")

    if amount_per_gesture < 0:
        raise ValueError("Negative number has been given as input for the desired amount per gesture.")

    # Specify the desired number of images for each gesture
    desired_amount = {gesture: amount_per_gesture for gesture in gestures_list}

    # Initialize the dictionary of current number of occurrences per each gesture
    current_amount = {gesture: 0 for gesture in gestures_list}

    # Initialize folder names, initialize paths dictionary
    data_dir = os.path.join(script_directory, "Data")
    example_dir = os.path.join(script_directory, "Examples")
    paths = {}

    # Create Data and Example folders if they do not not exist yet
    new_folder(data_dir, verbose=True)
    new_folder(example_dir, verbose=True)

    for gesture in gestures_list:

        # Create a subfolder per each gesture if it does not exist yet and add it to paths
        new = os.path.join(data_dir, gesture)
        new_folder(new)
        paths[gesture] = new

        # If the subfolder exists, make sure that the ordering is correct and
        # shift it if any skips are present
        # (e.g. "A_1.jpg", "A_2.jpg", ... instead of "A_1.jpg", "A_3.jpg", ...)
        repair_padding(new)

        # Since the directory is now correctly sorted by padding, we can read the current amounts
        files = os.listdir(new)
        files.sort(key=lambda file: int(re.split(r"[_|.]", file)[1]))
        current_amount[gesture] = 0 if not files else int(re.split(r"[_|.]", files[-1])[1])

        # Create a dummy example image for the current gesture if it does not have an example image yet
        new = os.path.join(example_dir, gesture + ".jpg")
        if not os.path.exists(new):
            cv2.imwrite(f"{new}", np.ones((540, 960)) * 255)
            warnings.warn(f"\nThe current gesture ({gesture}) does not have an example image yet, a dummy image has been created instead.")

    return data_dir, example_dir, desired_amount, current_amount, paths


def create_rectangle(origin, size_x, size_y):
    """
    Function to create the rectangle shape with given properties

    Parameters:
        origin: tuple of int, int
            Upper left corner of the rectangle.
        size_x: int
            Width of the rectangle
        size_y: int
            Height of the rectangle

    Returns:
        rectangle: list of 4 tuples of (int, int)
            The coordinates for the upper left, upper right, lower left and lower right corner
    """
    if not isinstance(origin, tuple):
        raise ValueError("Different datatype than tuple has been given as input for the origin.")
    if len(origin) != 2:
        raise ValueError("Different dimension than 2 given for the origin of the rectangle.")
    for val in origin:
        if not isinstance(val, int):
            raise ValueError("Different datatype than integer has been given for the coordinates of the origin.")
        if val < 0:
            raise ValueError("The coordinates for the origin cannot be negative.")

    if not isinstance(size_x, int) or not isinstance(size_y, int):
        raise ValueError("The height and width of the rectangle must be integers.")
    if size_x < 0 or size_y < 0:
        raise ValueError("The height and width of the rectangle must be positive integers.")

    origin_x, origin_y = origin
    upper_right, lower_left, lower_right = (origin_x + size_x, origin_y), (origin_x, origin_y + size_y), (origin_x + size_x, origin_y + size_y)
    return [origin, upper_right, lower_left, lower_right]


def get_dictionary(translations):
    """
    Function to create a dictionary of Czech-English pairs for gestures

    Parameters:
        translations: str
            Name of the txt file with translations of gestures in the following format: gesture_English \t gesture_Czech.

    Returns:
        dictionary: dict of gesture_English: gesture_Czech pairs
    """
    if not isinstance(translations, str):
        raise ValueError("Different datatype than string has been given for the name of the folder with translations.")
    if not os.path.exists(translations):
        raise ValueError("The given directory for the translations does not exist, please specify a correct directory path.")

    dictionary = {}
    with open(translations, "r") as pairs:
        for pair in pairs:
            pair = pair.rstrip("\n")
            english, czech = pair.split("\t")
            dictionary[english] = czech

    return dictionary


def parse_preprocessing(entry):
    """
    Function to parse a line specifying a layer that should be added
    to the tensorflow.keras.Sequential model.
    Inputs given in correct formats are assumed. All possible arguments must be entered.

    Parameters:
        entry: str
            A line specifying the given layer and its parameters.
            Possible layer names & their parameters:
                Blur blurring_type kernel_size sigma
                Threshold thresholding_type block_size constant
                Rescaling
                Grayscale

    Returns:
        layer: tensorflow.keras.layers
            An instance of the keras layers module
    """
    if not isinstance(entry, str):
        raise ValueError("Given input for preprocessing parsing is not a string.")

    entry_parsed = entry.split(" ")
    layer_name = entry_parsed[0]

    if layer_name == "Blurring":
        return Blurring(*entry_parsed[1:])
    if layer_name == "Threshold":
        return AdaptiveThresholding(*entry_parsed[1:])
    if layer_name == "Rescaling":
        return layers.Rescaling(scale=(1. / 255))
    if layer_name == "Grayscale":
        return Grayscale()
    return None
