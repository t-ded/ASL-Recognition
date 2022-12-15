# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 16:54:09 2022

@author: tomas
"""

import os
import re
import cv2
import numpy as np
import time
import warnings

script_dir = os.path.dirname("Image Collection.ipynb")
data_dir = os.path.join(script_dir, "Data")
example_dir = os.path.join(script_dir, "Examples")

# Specify the list of gestures, a subfolder will be created for each one
gestures = ["1", "2", "3", "A", "B"]

# Specify the desired number of images for each gesture
desired_amount = {"1": 200, "2": 200, "3": 200, "A": 500, "B": 1000}

# Initialize the dictionary of current number of occurrences per each gesture
current_amount = {gesture: 0 for gesture in gestures}

# Create Data folder if it does not exist yet
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# Create Examples folder if it does not exist yet
if not os.path.exists(example_dir):
    os.makedirs(example_dir)

for gesture in gestures:

    # Create a subfolder per each gesture if it does not exist yet
    new = os.path.join(data_dir, gesture)
    if not os.path.exists(new):
        os.makedirs(new)

    # If the subfolder exists, make sure that the ordering is correct and
    # shift it if any skips are present
    # (e.g. "A_1.jpg", "A_2.jpg", ... instead of "A_1.jpg", "A_3.jpg", ...)
    else:
        files = os.listdir(new)
        files.sort(key=lambda file: int(re.split(r"[_|.]", file)[1]))
        l = len(files)

        # Go through each file and if the run order skips a count, shift the respective file's run order
        for i in range(l - 1):
            name_split = re.split(r"[_|.]", files[i])
            name_split_next = re.split(r"[_|.]", files[i + 1])
            if (int(name_split[1]) + 1) != int(name_split_next[1]):
                new_name = name_split[0] + "_" + str(int(name_split[1]) + 1) + "." + name_split[2]
                os.rename(os.path.join(new, files[i + 1]), os.path.join(new, new_name))
                files = os.listdir(new)
                files.sort(key=lambda file: int(re.split(r"[_|.]", file)[1]))

        # Since the gesture subfolder is sorted by padding, we can use the last element as the current run
        current_amount[gesture] = 0 if not files else int(re.split(r"[_|.]", files[-1])[1])

    # Create blank example for each gesture if the example has not been added yet
    new = os.path.join(example_dir, gesture + ".jpg")
    if not os.path.exists(new):
        cv2.imwrite(f"{new}", np.ones((540, 960)) * 255)

paths = {gesture: os.path.join(data_dir, gesture) for gesture in gestures}

# The rectangle in the frame that is cropped from the web camera image
rect = [(225, 275), (425, 275),
        (225, 475), (425, 475)]


def image_capturing(method="adaptive", binary_threshold=0, last_mask=-1):
    """Capture image from the camera, convert it to grayscale
    and then perform preprocessing and thresholding.

    Keyword arguments:
    method (string) -- hand segmentation method selection from (default adaptive)
        mask = use background substraction based on mask taken at the beginning
        adaptive = use adaptive thresholding methods
    binary_thresh (integer) -- the thresholding parameter for binary thresholding in mask method (default 150)
    last_mask (integer) -- the last frame to consider for the creation of background mask in mask method
    """
    # Input management
    if method not in ["adaptive", "mask"]:
        raise ValueError("Method other than adaptive or mask has been given as input")

    if not isinstance(binary_threshold, int):
        raise ValueError("Different datatype than integer has been given as input for binary threshold")

    if not isinstance(last_mask, int):
        raise ValueError("Different datatype than integer has been given as input for last frame of background mask")

    if method == "mask" and binary_threshold == 0:
        wrn = "\nMask thresholding method has been chosen but binary threshold has not been specified."
        wrn += "\nThis will result in a fully one-colored image, please specify the threshold"
        warnings.warn(wrn)

    if method == "mask" and (binary_threshold < 0 or binary_threshold >= 255):
        wrn = "\nMask thresholding method has been chosen but binary threshold is under 0 or over 255."
        wrn += "\nThis will result in a fully one-colored image, please specify different threshold (between 1 and 254)"
        warnings.warn(wrn)

    if method == "mask" and last_mask <= 10:
        if last_mask == -1:
            wrn = "\nBackground substraction thresholding method has been chosen but last mask frame has not been specified."
        else:
            wrn = "\nBackground substraction thresholding method has been chosen but last mask frame is very low (<= 10)."
        wrn += "\nThis may result in unpredictable behaviour and possibly crash of the whole script, please correct the parameter"
        warnings.warn(wrn)

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    # Encapsulate the whole process to be able to close cameras in case of error
    try:

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

        # Initialize variables for background substraction
        frame_count = 0
        background = None

        # Perform the data collecting process for each gesture in the given gesture list
        for gesture in gestures:

            # Initialize necessary variables (different per gesture)
            current = current_amount[gesture] + 1
            counter = current
            end = desired_amount[gesture]
            flag = 0  # To know when a new gesture is being taken for the first time
            exit = 0  # To let the user end the process early by clicking the "Esc" key

            # Continue until the respective subfolder has the designated number of samples
            while counter <= end:
                ret, frame = cap.read()

                # Check validity and avoid mirroring if frame is present
                if not ret:
                    print("There has been a problem retrieving your frame")
                    break
                else:
                    frame_count += 1
                    frame = cv2.flip(frame, 1)

                # End the process for the current gesture in case the "q" key is hit
                key = cv2.waitKey(1)
                if key == ord("q"):
                    break

                # End the whole process in case the "Esc" key is hit
                if key == ord("\x1b"):
                    exit = 1
                    break

                # Create grayscale version
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Binarize the grayscale image using thresholding (with various methods)
                frame_binary = frame_gray[(rect[0][1] + 2):(rect[2][1] - 2),
                                          (rect[0][0] + 2):(rect[1][0] - 2)]

                # Version using background mask
                if method == "mask":
                    # Create mask for the background
                    if background is None:
                        background = frame_binary.copy().astype("float")
                    else:
                        if frame_count <= last_mask:
                            background = cv2.accumulateWeighted(frame_binary, background, 0.5)
                        # Use the mask for background substraction
                        else:
                            frame_binary = cv2.absdiff(background.astype("uint8"), frame_binary)
                            frame_binary = cv2.GaussianBlur(frame_binary, (3, 3), 0)
                            frame_binary = cv2.threshold(frame_binary, binary_threshold, 255, cv2.THRESH_BINARY)[1]

                # Version using adaptive thresholding
                elif method == "adaptive":
                    # For now (fastNlMeansDenoising (5, 15, 7) + 2 median blurs (5) + adaptive thresholding (3, 1))
                    frame_binary = cv2.fastNlMeansDenoising(frame_binary, None, 5, 15, 7)
                    frame_binary = cv2.medianBlur(frame_binary, 3)
                    frame_binary = cv2.GaussianBlur(frame_binary, (3, 3), 0)
                    frame_binary = cv2.adaptiveThreshold(frame_binary, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                                         cv2.THRESH_BINARY_INV, 3, 2)
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                    frame_binary = cv2.morphologyEx(frame_binary, cv2.MORPH_CLOSE, kernel)

                # Show all images

                # Live view with frame and text
                cv2.rectangle(frame, rect[0], rect[3], (0, 255, 0), 2)
                # acc = 0.0
                # txt = gesture.capitalize() + f" ({str(round(acc, 2))} %)"      # in preparation for model version
                txt = gesture.capitalize()
                cv2.putText(frame, txt, (rect[0][0], rect[0][1] - 15),
                            cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2)
                cv2.imshow("Camera view", frame)

                # Grayscale version
                cv2.imshow("Grayscale view", cv2.resize(frame_gray, (480, 360)))

                # Binary version
                if frame_count > last_mask:
                    cv2.imshow("Binary view", cv2.resize(frame_binary, (480, 360)))

                # Show example on new gesture
                if not flag:
                    example = cv2.imread(f"{os.path.join(example_dir, gesture)}" + ".jpg")
                    cv2.imshow("Example", cv2.resize(example, (380, 270)))
                    time.sleep(1)
                    flag = 1

                # To reduce the number of almost identical frames, only save every n frames
                if not current % 4:

                    # Create the naming for the file with the desired padding, i.e. ("gesture_run-number.jpg")
                    img_name = gesture + "_" + str(counter) + ".jpg"
                    img_path = r"%s" % os.path.join(paths[gesture], img_name)

                    # Save the cropped rectangle from the frame
                    if not cv2.imwrite(img_path,
                                       frame_binary):
                        print("Something went wrong during this attempt:",
                              f"gesture - {gesture}, run - {counter}")

                    counter += 1

                current += 1

            if exit:
                break

        cap.release()
        cv2.destroyAllWindows()
        return

    # Close the camera and all windows in case of unexpected fatality
    except:
        print("A fatality has occured, the program will now terminate")
        cap.release()
        cv2.destroyAllWindows()
        return


# image_capturing(method="adaptive")
image_capturing(method="mask", binary_threshold=50, last_mask=30)
