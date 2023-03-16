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
from tensorflow import expand_dims, convert_to_tensor, uint8
from utils import create_rectangle, new_folder
from model.model import build_preprocessing


def showcase_preprocessing(inp_shape):
    """
    Function for image capturing and showcasing & saving various preprocessing
    pipelines for comparison.

    Throughout the run of the function, you can use the following commands by using keys on your keyboard:
        Esc - terminate the whole process
        spacebar - move the rectangle into the other position (one should be more comfortable for fingerspelling)
        q - save the current frames for all the pipelines as well as their summaries into a seaprate txt file

    Parameters:
        inp_shape: list of ints
            Dimensions of the input (excluding the size of the batch).
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

        # Set a list of instructions
        instructions = ["I,G,B-tm-k7,T-tm-b5-c(3)",
                        "I,G,B-tm-k7,T-tg-b5-c(3)",
                        "I,G,B-tm-k7,T-tm-b7-c(3)",
                        "I,G,B-tm-k7,T-tg-b7-c(3)",
                        "I,G,B-tg-k7-s1,T-tm-b5-c(3)",
                        "I,G,B-tg-k7-s1,T-tg-b5-c(3)"]

        # Setting up preprocessing sequential pipelines
        pipelines_list = [build_preprocessing(inp_shape=inp_shape,
                                              instructions=instruction,
                                              name=f"Preprocessing_pipeline_{i + 1}")
                          for i, instruction in enumerate(instructions)]

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

            # Create rectangle cut
            frame_cut = frame[(rect[0][1] + 2):(rect[2][1] - 2),
                              (rect[0][0] + 2):(rect[1][0] - 2)]
            frame_cut_tensor = expand_dims(convert_to_tensor(frame_cut,
                                                             dtype=uint8),
                                           axis=0)

            # Obtain results from different preprocessing pipelines
            results_list = [pipeline(frame_cut_tensor) for pipeline in pipelines_list]

            # Live view with frame
            cv2.rectangle(frame, rect[0], rect[3], (0, 255, 0), 2)
            cv2.imshow("Camera view", frame)

            # Show results for all pipelines
            for i, result in enumerate(results_list):
                cv2.imshow(f"Preprocessing pipeline {i + 1}",
                           cv2.resize(result.numpy()[0, :, :, :], (320, 240)))

            # Save the current layout in case the "q" key is hit
            if key == ord("q"):

                # Set up the parent folder for this experiment and add preprocessing information
                if not save_counter:
                    new_folder("data_pipelines")
                    current_name = f"data_pipelines/experiment_{len(os.listdir('data_pipelines')) + 1}"
                    new_folder(current_name)

                    # Save pipeline architecture for each of them
                    with open(current_name + "\\pipeline_summaries.txt", "a+") as file:
                        for pipeline in pipelines_list:
                            pipeline.summary(print_fn=lambda x: file.write(x + "\n"))
                            file.write("\n")
                            for layer in pipeline.layers:
                                file.write(str(layer.get_config()) + "\n")
                            file.write("\n" * 3)

                # Set up the folder for a new save, save coloured version too
                save_name = current_name + f"/save_{save_counter + 1}"
                new_folder(save_name)
                coloured_path = r"%s" % os.path.join(save_name,
                                                     "coloured.jpg")
                if not cv2.imwrite(coloured_path,
                                   frame_cut_tensor.numpy()[0, :, :, :]):
                    print("Something went wrong during this save attempt:",
                          f"coloured_{save_counter + 1}")

                # Create the naming for the files with the desired padding
                for i, result in enumerate(results_list):
                    img_name = "pipeline" + "_" + str(i + 1) + "_" + instructions[i] + ".jpg"
                    img_path = r"%s" % os.path.join(save_name, img_name)

                    # Save each pipeline's result
                    if not cv2.imwrite(img_path, result.numpy()[0, :, :, :]):
                        print("Something went wrong during this save attempt:",
                              f"run - pipeline {i + 1}")

                save_counter += 1

    # Close the camera and all windows in case of unexpected fatality
    finally:
        cap.release()
        cv2.destroyAllWindows()
