"""
Dataset collection script for the Bachelor's Thesis project
on the following topic:
    "Construction of a Neural Networks model for translation of recorded sign language".
Turns on the camera and guides the user through the data collection process.

@author: Tomáš Děd
"""

import os
import re
import cv2
from utils import create_rectangle, get_dictionary


def collect_data(gesture_list, examples="Examples", data_directory="Data",
                 current_amounts=None, desired_amounts=None, gesture_paths=None,
                 translations="translations.txt", img_size=196):
    """
    Function for image capturing. Images in the rectangle you see on the screen
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
        data_directory: str (default "Data")
            Name of the folder in which to save the images.
        current_amounts: dict of str: int pairs (default None)
            Dictionary of current amounts of datapoints per each gesture.
        desired_amounts: dict of str: int pairs (default None)
            Dictionary of desired amounts of datapoints per each gesture.
        gesture_paths: dict of str: str pairs (default None)
            Dictionary of paths for each gesture.
        translations: str (default "translations.txt")
            Name of the txt file with translations of gestures in the following format: gesture_English \t gesture_Czech.
        img_size: int (default 196)
            Size of images for prediction.
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

    if not isinstance(img_size, int):
        raise ValueError("Different datatype than int has been given for the image size.")
    if img_size < 1:
        raise ValueError("Image size must be positive.")

    # The rectangle in the frame that is cropped from the web camera image
    # (one for torso location, one for fingerspelling location)
    rect_torso = create_rectangle((225, 275), img_size + 4, img_size + 4)
    rect_fingerspell_1 = create_rectangle((50, 50), img_size + 4, img_size + 4)
    rect_fingerspell_2 = create_rectangle((400, 50), img_size + 4, img_size + 4)
    rect = rect_torso

    # Encapsulate the whole process to be able to close cameras in case of error
    try:

        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        # Establish the windows and place them accordingly
        cv2.namedWindow("Camera view")
        cv2.resizeWindow("Camera view", 1080, 720)
        cv2.moveWindow("Camera view", 15, 200)

        cv2.namedWindow("Example")
        cv2.resizeWindow("Example", 640, 480)
        cv2.moveWindow("Example", 1125, 200)

        lang = True  # To let the user change language, True stands for English, False for Czech
        rectangle_position = 0  # Which position of the rectangle to use

        # Perform the data collecting process for each gesture in the given gesture list
        for gesture in gesture_list:

            # Initialize necessary variables (different per gesture)
            current = current_amounts[gesture] + 1
            end = desired_amounts[gesture]
            counter = current

            flag = 0  # To know when a new gesture is being taken for the first time
            exit_flag = 0  # To let the user end the process early by clicking the "Esc" key

            # Continue until the respective subfolder has the designated number of samples
            while counter <= end:
                ret, frame = cap.read()

                # Check validity and avoid mirroring if frame is present
                if not ret:
                    print("There has been a problem retrieving your frame")
                    print("Try adjusting the camera number in specification of cap (default 0)")
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

                # Live view with frame and text (colorcoded to indicate whether images are being saved)
                if current - current_amounts[gesture] > 75:
                    cv2.rectangle(frame, rect[0], rect[3], (0, 255, 0), 2)
                else:
                    cv2.rectangle(frame, rect[0], rect[3], (0, 0, 255), 2)
                txt = gesture.capitalize()
                if not lang:
                    txt = dictionary[txt]
                if current - current_amounts[gesture] > 75:
                    cv2.putText(frame, txt, (rect[0][0], rect[0][1] - 15),
                                cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, txt, (rect[0][0], rect[0][1] - 15),
                                cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)
                cv2.imshow("Camera view", frame)

                # Show example on new gesture
                if not flag:
                    example = cv2.imread(f"{os.path.join(examples, gesture)}" + ".jpg")
                    cv2.imshow("Example", cv2.resize(example, (640, 480)))
                    flag = 1

                    # Show big example to warn about new gesture and enable readjustment
                    cv2.namedWindow("BigExample")
                    cv2.resizeWindow("BigExample", 1280, 720)
                    cv2.moveWindow("BigExample", 125, 50)
                    cv2.imshow("BigExample", cv2.resize(example, (1280, 720)))
                    cv2.waitKey(2000)
                    cv2.destroyWindow("BigExample")

                # To reduce the number of almost identical frames, only save every n frames
                # To give space for adjustments and "learning" a new sign, only start collecting after some time
                if not current % 10 and current - current_amounts[gesture] > 80:

                    # Create the naming for the file with the desired padding, i.e. ("gesture_run-number.jpg")
                    img_name = gesture + "_" + str(counter) + ".jpg"
                    img_path = r"%s" % os.path.join(gesture_paths[gesture], img_name)

                    # Save the cropped rectangle from the frame
                    if not cv2.imwrite(img_path,
                                       cv2.resize(frame[(rect[0][1] + 2):(rect[2][1] - 2),
                                                        (rect[0][0] + 2):(rect[1][0] - 2)],
                                                  (img_size, img_size))):
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
