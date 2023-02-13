# -*- coding: utf-8 -*-
# disable=C0301, C0103, E1101
"""
File with multiple functions for the purpose of Bachelor's thesis
on the following topic:
    "Construction of a Neural Networks model for translation of recorded sign language"
"""

import os
import re
import warnings
import cv2
import numpy as np
from tensorflow.keras.models import Sequential


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


def create_rectangle(origin, size_X, size_Y):
    """
    Function to create the rectangle shape with given properties

    Parameters:
        origin: tuple of int, int
            Upper left corner of the rectangle.
        size_X: int
            Width of the rectangle
        size_Y: int
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

    if not isinstance(size_X, int) or not isinstance(size_Y, int):
        raise ValueError("The height and width of the rectangle must be integers.")
    if size_X < 0 or size_Y < 0:
        raise ValueError("The height and width of the rectangle must be positive integers.")

    X, Y = origin
    upper_right, lower_left, lower_right = (X + size_X, Y), (X, Y + size_Y), (X + size_X, Y + size_Y)
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


def image_capturing(gesture_list, examples="Examples", save=True, predict=False,
                    data_directory=None, current_amounts=None, desired_amounts=None,
                    gesture_paths=None, model=None, translations="translations.txt"):
    """
    Function for image capturing. Enables usage as data collector or purely as image obtainer.
    If the save parameter is True, thresholded binary images in the rectangle you see on the screen
    will be saved in appropriate directory for each gesture.

    Throughout the run of the function, you can use the following commands by using keys on your keyboard:
        Esc - terminate the whole process
        q - skip the current gesture and move to the next one
        l - switch language from English to Czech and vice versa
        spacebar - move the rectangle into the other position (one should be more comfortable for fingerspelling)

    Parameters:
        gesture_list: list of str
            List of gestures to go through
        examples: str (default "Examples")
            Name of the folder with examples.
        save: bool (default True)
            Whether or not to save the captured images, i.e. whether or not to collect new data.
            If True, legitimate directory for saving needs to be specified.
        predict: bool (default False)
            Whether or not to include prediction on the screen
        data_directory: str (default None)
            Name of the folder in which to save the images in case save is True.
        current_amounts: dict of str: int pairs (default None)
            Dictionary of current amounts of datapoints per each gesture. Only relevant in case save is True.
        desired_amounts: dict of str: int pairs (default None)
            Dictionary of desired amounts of datapoints per each gesture. Only relevant in case save is True.
        gesture_paths: dict of str: str pairs (default None)
            Dictionary of paths for each gesture. Only relevant in case save is True.
        model: keras.engine.sequential.Sequential (default None)
            Model that the user would like to use for prediction.
        translations: str (default "translations.txt")
            Name of the txt file with translations of gestures in the following format: gesture_English \t gesture_Czech.
    """
    # Input management
    if not isinstance(gesture_list, list):
        raise ValueError("Different datatype than list has been given as input for the list of gestures.")
    for val in gesture_list:
        if not isinstance(val, str):
            raise ValueError("Different datatype than string has been given for one of the values in the list of gestures.")
    if len(gesture_list) != len(set(gesture_list)):
        raise ValueError("There are some gesture duplicates in the list of gestures, please check.")

    if not isinstance(examples, str):
        raise ValueError("Different datatype than string has been given for the name of the folder with examples.")
    if not os.path.exists(examples):
        raise ValueError("The given directory for the examples does not exist, please specify a correct directory path.")
    files = os.listdir(examples)
    if len(files) != len(gesture_list):
        raise ValueError("The length of the example folder does not correspond to the list of gestures, please adjust.")
    gesture_names = [re.split(r"[.]", file)[0] for file in files]
    if len(gesture_names) != len(set(gesture_names)):
        raise ValueError("There are some gesture duplicates in the example folder, please check.")
    if set(gesture_names) != set(gesture_list):
        raise ValueError("The list of gestures does not correspond to the files in the given example folder.")

    if not isinstance(save, bool):
        raise ValueError("Different datatype than boolean has been given as input for the save parameter.")

    if save:
        if not isinstance(data_directory, str):
            raise ValueError("Different datatype than string has been given for the name of the folder to save images in.")
        if not os.path.exists(data_directory):
            raise ValueError("The given directory to save images does not exist, please specify a correct directory path.")

        if not isinstance(current_amounts, dict):
            raise ValueError("Different datatype than dictionary has been given for the current_amounts parameter.")
        for key, val in current_amounts.items():
            if not isinstance(key, str) or not isinstance(val, int):
                raise ValueError("Different datatype than string for key or integer for value has been given for one of the values in the current_amounts dictionary.")
            if val < 0:
                raise ValueError("Negative value has been given as current amount of datapoints for one of the gestures.")

        if not isinstance(desired_amounts, dict):
            raise ValueError("Different datatype than dictionary has been given for the desired_amounts parameter.")
        for key, val in desired_amounts.items():
            if not isinstance(key, str) or not isinstance(val, int):
                raise ValueError("Different datatype than string for key or integer for value has been given for one of the values in the desired_amounts dictionary.")

    if not isinstance(predict, bool):
        raise ValueError("Different datatype than boolean has been given as input for the predict parameter.")

    if predict:
        if not model:
            raise ValueError("Cannot perform predictions without a model specified.")
        if not isinstance(model, Sequential):
            raise ValueError("Different datatype than keras.engine.sequential.Sequential has been given as an input for the model parameter.")
        if model.layers[-1].units != len(gesture_list):
            warnings.warn("Given model has different output size than given gesture list so the predictions might be incorrect.")
        if not data_directory:
            raise ValueError("Please assign a data directory to provide a list of labels for prediction")

    if translations is not None:
        if not isinstance(translations, str):
            raise ValueError("Different datatype than string has been given for the name of the folder with translations.")
        if not os.path.exists(translations):
            raise ValueError("The given directory for the translations does not exist, please specify a correct directory path.")
        dictionary = get_dictionary(translations)
        if len(dictionary) != len(gesture_list):
            raise ValueError("The length of the translations list does not correspond to the list of gestures, please adjust.")
        if set(dictionary.keys()) != set(gesture_list):
            raise ValueError("The list of gestures does not correspond to the gestures in the given list of translations.")

    # The rectangle in the frame that is cropped from the web camera image
    # (one for torso location, one for fingerspelling location)
    rect_torso = create_rectangle((225, 275), 200, 200)
    rect_fingerspell_1 = create_rectangle((50, 50), 200, 200)
    rect_fingerspell_2 = create_rectangle((400, 50), 200, 200)
    rect = rect_torso

    # Encapsulate the whole process to be able to close cameras in case of error
    try:

        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        # Establish the windows and place them accordingly
        cv2.namedWindow("Camera view")
        cv2.resizeWindow("Camera view", 640, 480)
        cv2.moveWindow("Camera view", 15, 200)

        cv2.namedWindow("Grayscale view")
        cv2.resizeWindow("Grayscale view", 480, 360)
        cv2.moveWindow("Grayscale view", 655, 30)

        cv2.namedWindow("Binary view")
        cv2.resizeWindow("Binary view", 480, 360)
        cv2.moveWindow("Binary view", 655, 430)

        cv2.namedWindow("Example")
        cv2.resizeWindow("Example", 380, 270)
        cv2.moveWindow("Example", 1125, 280)

        if predict:
            gestures_folder = os.listdir(data_directory)

        lang = True  # To let the user change language, True stands for English, False for Czech
        rectangle_position = 0  # Which position of the rectangle to use

        # Perform the data collecting process for each gesture in the given gesture list
        for gesture in gesture_list:

            # Initialize necessary variables (different per gesture)
            # There is no limit on when to end during the process of not saving - has to be commanded manually
            if save:
                current = current_amounts[gesture] + 1
                end = desired_amounts[gesture]
            else:
                current = 0
                end = np.inf
            counter = current

            flag = 0  # To know when a new gesture is being taken for the first time
            exit_flag = 0  # To let the user end the process early by clicking the "Esc" key

            # Continue until the respective subfolder has the designated number of samples
            while counter <= end:
                ret, frame = cap.read()

                # Check validity and avoid mirroring if frame is present
                if not ret:
                    print("There has been a problem retrieving your frame")
                    break
                frame = cv2.flip(frame, 1)

                # End the process for the current gesture in case the "q" key is hit
                key = cv2.waitKey(1)
                if key == ord("q"):
                    break

                # End the whole process in case the "Esc" key is hit
                if key == ord("\x1b"):
                    exit_flag = 1
                    break

                # Change language settings in case the "l" key is hit
                if key == ord("l"):
                    lang = not lang

                # Change the rectangle position if the "spacebar" key is hit
                if key == ord(" "):
                    rectangle_position += 1
                    rect = [rect_torso, rect_fingerspell_1, rect_fingerspell_2][rectangle_position % 3]

                # Create grayscale version
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Binarize the grayscale image using adaptive thresholding
                frame_binary = frame_gray[(rect[0][1] + 2):(rect[2][1] - 2),
                                          (rect[0][0] + 2):(rect[1][0] - 2)]

                # Preprocessing, denoising, blurring
                frame_binary = cv2.fastNlMeansDenoising(frame_binary, None, 5, 15, 7)
                frame_binary = cv2.medianBlur(frame_binary, 3)
                frame_binary = cv2.GaussianBlur(frame_binary, (3, 3), 0)
                # Adaptive thresholding
                frame_binary = cv2.adaptiveThreshold(frame_binary, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                                     cv2.THRESH_BINARY_INV, 3, 2)
                # Closing operation on the thresholded image
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                frame_binary = cv2.morphologyEx(frame_binary, cv2.MORPH_CLOSE, kernel)

                # Show all images

                # Live view with frame and text
                if (current - current_amounts[gesture] > 65):
                    cv2.rectangle(frame, rect[0], rect[3], (0, 255, 0), 2)
                else:
                    cv2.rectangle(frame, rect[0], rect[3], (0, 0, 255), 2)
                # Add information about prediction if expected, otherwise just show the name of the gesture
                if predict:
                    prediction = model(np.expand_dims(np.expand_dims(cv2.resize(frame_binary,
                                                                                (196, 196)),
                                                                     axis=0), axis=-1),
                                       training=False).numpy()
                    txt = gestures_folder[np.argmax(prediction, axis=1)[0]]
                else:
                    txt = gesture.capitalize()
                if not lang:
                    txt = dictionary[txt]
                if (current - current_amounts[gesture] > 65):
                    cv2.putText(frame, txt, (rect[0][0], rect[0][1] - 15),
                                cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, txt, (rect[0][0], rect[0][1] - 15),
                                cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)
                cv2.imshow("Camera view", frame)

                # Grayscale version
                cv2.imshow("Grayscale view", cv2.resize(frame_gray, (480, 360)))

                # Binary version
                cv2.imshow("Binary view", cv2.resize(frame_binary, (480, 360)))

                # Show example on new gesture
                if not flag:
                    example = cv2.imread(f"{os.path.join(examples, gesture)}" + ".jpg")
                    cv2.imshow("Example", cv2.resize(example, (380, 270)))
                    flag = 1

                    # Show big example to warn about new gesture and enable readjustment
                    if save:
	                    cv2.namedWindow("BigExample")
	                    cv2.resizeWindow("BigExample", 1280, 720)
	                    cv2.moveWindow("BigExample", 125, 50)
	                    cv2.imshow("BigExample", cv2.resize(example, (1280, 720)))
	                    cv2.waitKey(1500)
	                    cv2.destroyWindow("BigExample")
                    

                # To reduce the number of almost identical frames, only save every n frames
                # To give space for adjustments and "learning" a new sign, only start collecting after some time
                if save and not current % 3 and current - current_amounts[gesture] > 70:

                    # Create the naming for the file with the desired padding, i.e. ("gesture_run-number.jpg")
                    img_name = gesture + "_" + str(counter) + ".jpg"
                    img_path = r"%s" % os.path.join(gesture_paths[gesture], img_name)

                    # Save the cropped rectangle from the frame
                    if not cv2.imwrite(img_path,
                                       cv2.resize(frame[(rect[0][1] + 2):(rect[2][1] - 2),
                                                        (rect[0][0] + 2):(rect[1][0] - 2)],
                                                  (196, 196))):
                        print("Something went wrong during this attempt:",
                              f"gesture - {gesture}, run - {counter}")

                    counter += 1

                current += 1

            if exit_flag:
                break

    # Close the camera and all windows in case of unexpected fatality
    finally:
        cap.release()
        cv2.destroyAllWindows()
