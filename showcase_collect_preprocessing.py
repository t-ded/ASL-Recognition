"""
Preprocessing showcasing and image collecting script for the Bachelor's Thesis project
on the following topic:
    "Construction of a Neural Networks model for translation of recorded sign language".
Turns on the camera, presents the camera view, various versions of preprocessing pipeline
and then possibly saves these into given folders for comparison.

@author: Tomáš Děd
"""

import os
import cv2
from utils import create_rectangle, new_folder
from tensorflow.keras import layers, Sequential
from keras_cv.layers import Grayscale
from model.preprocessing import AdaptiveThresholding, Blurring


def showcase_preprocessing():
    """
    Function for image capturing and showcasing & saving various preprocessing
    pipelines for comparison.

    Throughout the run of the function, you can use the following commands by using keys on your keyboard:
        Esc - terminate the whole process
        spacebar - move the rectangle into the other position (one should be more comfortable for fingerspelling)
    """
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

        for i in range(6):
            cv2.namedWindow(f"Preprocessing pipeline {i + 1}")
            cv2.resizeWindow(f"Preprocessing pipeline {i + 1}", 320, 240)
            cv2.moveWindow(f"Preprocessing pipeline {i + 1}", 655 + 325 * (i % 2), 16 + 270 * (i // 2))

        # !!!!!!!!!!!!!!!!!!!!!
        # TO-DO
        # Setting up preprocessing sequential pipelines

        rectangle_position = 0  # Which position of the rectangle to use
        save_counter = 0  # Count the number of times user has requested saving pipelines' results

        # Continue until the user terminates the process
        while True:
            ret, frame = cap.read()

            # Check validity and avoid mirroring if frame is present
            if not ret:
                print("There has been a problem retrieving your frame")
                break
            frame = cv2.flip(frame, 1)

            # End the whole process in case the "Esc" key is hit
            key = cv2.waitKey(1)
            if key == ord("\x1b"):
                break

            # Change the rectangle position if the "spacebar" key is hit
            if key == ord(" "):
                rectangle_position += 1
                rect = [rect_torso, rect_fingerspell_1, rect_fingerspell_2][rectangle_position % 3]

            # Create grayscale version
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_gray = frame_gray[(rect[0][1] + 2):(rect[2][1] - 2),
                                    (rect[0][0] + 2):(rect[1][0] - 2)]

            # Try different preprocessing pipelines
            frame_binary_1 = frame_gray.copy()
            frame_binary_2 = frame_gray.copy()
            frame_binary_3 = frame_gray.copy()
            frame_binary_4 = frame_gray.copy()
            frame_binary_5 = frame_gray.copy()
            frame_binary_6 = frame_gray.copy()
            pipelines_list = [frame_binary_1, frame_binary_2, frame_binary_3,
                              frame_binary_4, frame_binary_5, frame_binary_6]

            # Live view with frame
            cv2.rectangle(frame, rect[0], rect[3], (0, 255, 0), 2)
            cv2.imshow("Camera view", frame)

            # Show results for all pipelines
            for i, pipeline in enumerate(pipelines_list):
                cv2.imshow(f"Preprocessing pipeline {i + 1}", cv2.resize(pipeline, (320, 240)))

            # Save the current layout in case the "q" key is hit
            if key == ord("q"):

                # Set up the parent folder for this experiment and add preprocessing information
                if not save_counter:
                    new_folder("data_pipelines")
# TO-DO: Might eventually add repair padding function for folders (aaa_1) -> use that here on "data_pipelines"
                    current_name = f"data_pipelines/experiment_{len(os.listdir('data_pipelines')) + 1}"
                    new_folder(current_name)
# TO-DO: Add saving information for this preprocessing run (e.g. json)

                # Set up the folder for a new save
                save_name = current_name + f"/save_{save_counter + 1}"
                new_folder(save_name)

                # Create the naming for the files with the desired padding
                for i, pipeline in enumerate(pipelines_list):
                    img_name = "pipeline" + "_" + str(i + 1) + ".jpg"
                    img_path = r"%s" % os.path.join(save_name, img_name)

                    # Save each pipeline's result and information
                    if not cv2.imwrite(img_path, pipeline):
                        print("Something went wrong during this attempt:",
                              f"run - pipeline {i}")

                save_counter += 1

    # Close the camera and all windows in case of unexpected fatality
    finally:
        cap.release()
        cv2.destroyAllWindows()


showcase_preprocessing()
