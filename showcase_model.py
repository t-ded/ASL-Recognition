"""
Model or process showcasing script for the Bachelor's Thesis project
on the following topic:
    "Construction of a Neural Networks model for translation of recorded sign language".
Turns on the camera and presents the camera view, the final preprocessing steps
and possibly the model predictions.

@author: Tomáš Děd
"""

import os
import re
import warnings
import numpy as np
import cv2
import tensorflow as tf
from utils import create_rectangle, get_dictionary


def showcase_model(gesture_list, examples="Examples", predict=False,
                   model=None, translations="translations.txt", img_size=196):
    """
    Function for image capturing and showcasing the process, preprocessing
    and possibly the model prediction if given the model.

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
        predict: bool (default False)
            Whether or not to include prediction on the screen
        model: keras.engine.sequential.Sequential (default None)
            Model that the user would like to use for prediction.
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

    if not isinstance(predict, bool):
        raise ValueError("Different datatype than boolean has been given as input for the predict parameter.")

    if predict:
        if not model:
            raise ValueError("Cannot perform predictions without a model specified.")
        if not isinstance(model, tf.keras.Model):
            raise ValueError("Different datatype than tf.keras.Model has been given as an input for the model parameter.")
        if model.get_layer(index=-1).output.shape[-1] != len(gesture_list):
            warnings.warn("\nGiven model has different output size than given gesture list so the predictions might be incorrect.\n")

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

    # Naive solution to allow looping through the gestures in prediction environment
    if predict:
        gesture_list *= 5

    # Encapsulate the whole process to be able to close cameras in case of error
    try:

        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        # Establish the windows and place them accordingly
        cv2.namedWindow("Camera view")
        cv2.resizeWindow("Camera view", 1080, 720)
        cv2.moveWindow("Camera view", 100, 150)

        cv2.namedWindow("Example")
        cv2.resizeWindow("Example", 480, 360)
        cv2.moveWindow("Example", 750, 230)

        lang = True  # To let the user change language, True stands for English, False for Czech
        rectangle_position = 0  # Which position of the rectangle to use

        # Perform the data collecting process for each gesture in the given gesture list
        for gesture in gesture_list:

            # Initialize necessary variables (different per gesture)
            flag = 0  # To know when a new gesture is being taken for the first time
            exit_flag = 0  # To let the user end the process early by clicking the "Esc" key

            # Continue until the user terminates the process
            while True:
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

                # Create rectangle cut
                frame_cut = frame[(rect[0][1] + 2):(rect[2][1] - 2),
                                  (rect[0][0] + 2):(rect[1][0] - 2)]

                # Show all images

                # Live view with frame and text
                cv2.rectangle(frame, rect[0], rect[3], (0, 255, 0), 2)
                # Add information about prediction if expected, otherwise just show the name of the gesture
                if predict:
                    prediction = model(frame_cut[None, :],
                                       training=False).numpy()
                    probability = prediction.max(axis=-1).round(2)
                    txt = gesture_list[np.argmax(prediction, axis=1)[0]]
                    if not lang:
                        txt = dictionary[txt]
                    txt += " (" + str(probability[0]) + ")"
                else:
                    txt = gesture.capitalize()
                    if not lang:
                        txt = dictionary[txt]
                cv2.putText(frame, txt, (rect[0][0], rect[0][1] - 15),
                            cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2)
                cv2.imshow("Camera view", frame)

                # Show example on new gesture
                if not flag:
                    example = cv2.imread(f"{os.path.join(examples, gesture)}" + ".jpg")
                    cv2.imshow("Example", cv2.resize(example, (480, 360)))
                    flag = 1

            if exit_flag:
                break

    # Close the camera and all windows in case of unexpected fatality
    finally:
        cap.release()
        cv2.destroyAllWindows()
