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
from timeit import default_timer
from itertools import cycle
from collections import Counter
import cv2
import tensorflow as tf
import pyttsx3
from utils import create_rectangle, get_dictionary


def showcase_model(gesture_list, examples="Examples",
                   predict=False, model=None,
                   translations="translations.txt", img_size=196,
                   guided=False, sequence_list=[]):
    """
    Function for image capturing and showcasing the process, preprocessing
    and possibly the model prediction if given the model.

    Throughout the run of the function, you can use the following commands by using keys on your keyboard:
        Esc - terminate the whole process
        q - skip the current gesture and move to the next one (+ voice prediction if sequence_list given)
        l - switch language from English to Czech and vice versa
        p - pause the process
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
        guided: bool (default False)
            Whether or not to assume the current example as the correct label
        sequence_list: list of str (default [])
            List of sequences to use for voice demonstration
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

    if not isinstance(guided, bool):
        raise ValueError("Different datatype than boolean has been given as input for the guided parameter.")

    if not isinstance(sequence_list, list):
        raise ValueError("Different datatype than list has been given as input for the sequence_list parameter.")
    if not all(isinstance(prompt, str) for prompt in sequence_list):
        raise ValueError("An element in the sequence_list parameter is not a string.")
    if not set(sequence_list).issubset(set(gesture_list)):
        raise ValueError("An element in the sequence_list parameter is not present in the list of gestures.")

    # The rectangle in the frame that is cropped from the web camera image
    # (one for torso location, one for fingerspelling location)
    rect_size = int(img_size * 1.25) + 4
    rect_torso = create_rectangle((225, 225), rect_size, rect_size)
    rect_fingerspell_1 = create_rectangle((50, 50), rect_size, rect_size)
    rect_fingerspell_2 = create_rectangle((350, 50), rect_size, rect_size)
    rect = rect_torso

    # Loop through the gesture list in prediction environment
    if predict:
        full_gestures = gesture_list
        gesture_list = cycle(gesture_list)

        # Change the gesture list if expected to voice and initialize the voice engine
        if sequence_list:
            gesture_list = sequence_list
            engine = pyttsx3.init()

    # Encapsulate the whole process to be able to close cameras in case of error
    try:

        cap = cv2.VideoCapture(0)

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

        # Perform the data collecting process for each gesture in the given gesture list
        for gesture in gesture_list:

            # Initialize necessary variables (different per gesture)
            flag = 0  # To know when a new gesture is being taken for the first time
            exit_flag = 0  # To let the user end the process early by clicking the "Esc" key
            if predict:
                preds_list = Counter(full_gestures)  # To voice the predictions if the respective prompt given
            frame_count = 0  # To ignore the first few frames for the voiced prediction

            # Continue until the user terminates the process
            while True:
                ret, frame = cap.read()

                # Check validity and avoid mirroring if frame is present
                if not ret:
                    print("There has been a problem retrieving your frame")
                    print("Try adjusting the camera number in specification of cap (default 0)")
                    break
                frame = cv2.flip(frame, 1)
                frame_count += 1

                # End the process for the current gesture in case the "q" key is hit
                key = cv2.waitKey(1)
                if key == ord("q"):
                    if sequence_list:
                        final_pred = preds_list.most_common(1)[0]
                        if final_pred[1] > 1:
                            engine.say(final_pred[0])
                            engine.runAndWait()
                        else:
                            print("Not enough data collected for voicing the prediction")
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

                # Create rectangle cut
                frame_cut = frame[(rect[0][1] + 2):(rect[2][1] - 2),
                                  (rect[0][0] + 2):(rect[1][0] - 2)]
                frame_cut = cv2.resize(frame_cut, (img_size, img_size))

                # Adjust the color of the text and frame based on the current state
                # Green - running model prediction, Orange - model prediction is paused
                # If 'guided' is set to True, then green corresponds to correct prediction
                # and red corresponds to incorrect prediction(s)
                if pause_flag:
                    color = (0, 128, 255)
                else:
                    color = (0, 255, 0)

                # Live view with frame and text (colorcoded as specified above)
                cv2.rectangle(frame, rect[0], rect[3], color, 2)
                # Add information about prediction if expected, otherwise just show the name of the gesture
                if predict and not pause_flag:

                    # Measure the time taken for the prediction
                    pred_time_start = default_timer()
                    prediction = model(frame_cut[None, :],
                                       training=False)
                    pred_time = round(default_timer() - pred_time_start, 3)

                    # Get k most confident predictions
                    most_confident = tf.math.top_k(prediction, k=3)
                    most_confident_indices = most_confident.indices.numpy()[0]
                    most_confident_probabilities = most_confident.values.numpy()[0].round(2)
                    most_confident_predictions = [full_gestures[ind] for ind in most_confident_indices]

                    # Display all of the most confident predictions
                    for i, (pred, prob) in enumerate(zip(most_confident_predictions, most_confident_probabilities)):
                        txt = dictionary[pred] if not lang else pred
                        txt += " (" + str(prob) + ")"

                        # If 'guided' is set to True, display predictions in red except
                        # the correct one (if present)
                        pred_color = color
                        if guided:
                            pred_color = (0, 0, 255)
                            if pred == gesture:
                                pred_color = color
                        cv2.putText(frame, txt, (5, 350 + i * 35),
                                    cv2.FONT_HERSHEY_DUPLEX, 0.8, pred_color, 2)

                        # If expected to voice the prediction, store the most confident one
                        # starting after some initial warmup period
                        if sequence_list and i == 0 and frame_count > 60:
                            preds_list.update([pred])

                    # Display average time per gesture
                    cv2.putText(frame, "TPG: " + str(pred_time) + " s", (5, 470),
                                cv2.FONT_HERSHEY_DUPLEX, 0.8, color, 2)
                else:
                    txt = gesture
                    if not lang:
                        txt = dictionary[txt]
                    cv2.putText(frame, txt, (rect[0][0], rect[0][1] - 15),
                                cv2.FONT_HERSHEY_DUPLEX, 1, color, 2)
                cv2.imshow("Camera view", cv2.resize(frame, (800, 600)))

                # Show example on new gesture
                if not flag:
                    example = cv2.imread(f"{os.path.join(examples, gesture)}" + ".jpg")
                    cv2.imshow("Example", cv2.resize(example, (640, 480)))
                    flag = 1

            if exit_flag:
                break

    # Close the camera and all windows in case of unexpected fatality
    finally:
        cap.release()
        cv2.destroyAllWindows()
