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
from timeit import default_timer
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
        p - pause the process
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
    rect_size = int(img_size * 1.25) + 4
    rect_torso = create_rectangle((225, 225), rect_size, rect_size)
    rect_fingerspell_1 = create_rectangle((50, 50), rect_size, rect_size)
    rect_fingerspell_2 = create_rectangle((350, 50), rect_size, rect_size)
    rect = rect_torso

    # Encapsulate the whole process to be able to close cameras in case of error
    try:

        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        # Establish the windows and place them accordingly
        cv2.namedWindow("Camera view")
        cv2.resizeWindow("Camera view", 800, 600)
        cv2.moveWindow("Camera view", 25, 150)

        cv2.namedWindow("Example")
        cv2.resizeWindow("Example", 640, 480)
        cv2.moveWindow("Example", 850, 230)

        lang = True  # To let the user change language, True stands for English, False for Czech
        rectangle_position = 0  # Which position of the rectangle to use
        pause_flag = False  # Enable the user to pause the process
        num_gestures = len(gesture_list)  # To display the progress

        # Perform the data collecting process for each gesture in the given gesture list
        for ind, gesture in enumerate(gesture_list):

            # Initialize necessary variables (different per gesture)
            current = current_amounts[gesture] + 1
            end = desired_amounts[gesture]
            counter = current

            flag = 0  # To know when a new gesture is being taken for the first time
            exit_flag = 0  # To let the user end the process early by clicking the "Esc" key
            eta = "N/A"  # To display the estimated time before the next gesture
            eta_flag = True  # Flag to enable estimation of duration per saved image

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

                # Pause the process if the "p" key is hit
                if key == ord("p"):
                    pause_flag = not pause_flag

                # Change the rectangle position if the "spacebar" key is hit
                if key == ord(" "):
                    rectangle_position += 1
                    rect = [rect_torso, rect_fingerspell_1, rect_fingerspell_2][rectangle_position % 3]

                # Adjust the color of the text and frame based on the current state
                # Green - saving images, Orange - image saving is paused, Red - not saving images
                if pause_flag:
                    color = (0, 128, 255)
                elif current - current_amounts[gesture] <= 75:
                    color = (0, 0, 255)
                else:
                    color = (0, 255, 0)

                # Live view with frame and text (colorcoded as specified above)
                cv2.rectangle(frame, rect[0], rect[3], color, 2)
                txt = gesture.capitalize()
                if not lang:
                    txt = dictionary[txt]
                cv2.putText(frame, txt, (rect[0][0], rect[0][1] - 15),
                            cv2.FONT_HERSHEY_DUPLEX, 1, color, 2)

                # Display the gesture progress as well as estimation until the next gesture
                cv2.putText(frame, f"{str(ind + 1)}/{str(num_gestures)}", (5, 435),
                            cv2.FONT_HERSHEY_DUPLEX, 0.8, color, 2)
                cv2.putText(frame, "ETA: " + eta, (5, 470),
                            cv2.FONT_HERSHEY_DUPLEX, 0.8, color, 2)

                cv2.imshow("Camera view", cv2.resize(frame, (800, 600)))

                # Show example on new gesture
                if not flag:
                    example = cv2.imread(f"{os.path.join(examples, gesture)}" + ".jpg")
                    cv2.imshow("Example", cv2.resize(example, (480, 360)))
                    flag = 1

                    # Show big example to warn about new gesture and enable readjustment
                    cv2.namedWindow("BigExample")
                    cv2.resizeWindow("BigExample", 1280, 720)
                    cv2.moveWindow("BigExample", 125, 50)
                    cv2.imshow("BigExample", cv2.resize(example, (1280, 720)))
                    cv2.waitKey(3000)
                    cv2.destroyWindow("BigExample")

                # Halt the process until the "p" key is hit again
                if pause_flag:
                    eta_flag = True
                    continue

                # To reduce the number of almost identical frames, only save every n frames
                # To give space for adjustments and "learning" a new sign, only start collecting after some time
                if not current % 10 and current - current_amounts[gesture] > 80:

                    # Estimate the time from the last saved image
                    if eta_flag:
                        eta_flag = not eta_flag
                        start = default_timer()
                    else:
                        eta_flag = not eta_flag
                        eta = str(round((default_timer() - start) * (end - counter), 1)) + " s"

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
