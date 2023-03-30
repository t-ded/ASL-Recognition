# -*- coding: utf-8 -*-
# disable=C0301, C0103, E1101
# TODO: Adjust README.md file
# TODO: Adjust the docstring for this run function
# TODO: Add BatchNormalization layer support to model.py
# TODO: Add BatchNormalization layers to default architecture in config.json
"""
A simple function to enable running
the different scripts from command line.

Mutually exclusive process specification parameters:

    -col (--collect)
        Start the process of data collection.
        Image size is specified in config.json file.
        Primarily uses the collect_dataset.py script.
    -prep (--preprocess)
        Start the demonstration for 6 different preprocessing pipelines.
        The pipelines are specified in showcase_collect_preprocessing.py.
    -tr (--train)
        Build a model based on given architecture, train the model and save it.
        The architecture and hyperparameters are either specified
        in the command line with this parameter or set to default and given
        by the config.json file.
        Primarily uses the model.py script.
    -show (--showcase)
        Demonstrate the script environment.
        This is the default procedure if none of procedure options is given.
        Primarily uses the showcase_model.py script.
    -pred (--predict)
        Start the camera and showcases the model prediction using pretrained model.
        The model is expected in the model/current directory.
        Primarily uses the showcase_model.py script.

Training parameters:
    TODO
"""

import argparse
import os
import re
import datetime
import json
import tensorflow as tf
import tensorflow_addons as tfa
import utils
from collect_dataset import collect_data
from showcase_collect_preprocessing import showcase_preprocessing
from showcase_model import showcase_model
from model.preprocessing import Grayscale, AdaptiveThresholding, Blurring, image_augmentation, ConfusionMatrixCallback
from model.model import build_model, build_preprocessing

parser = argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument("--config_dir", default="", type=str, help="Directory with the config.json file")

# Specify procedure
procedure = parser.add_mutually_exclusive_group()
procedure.add_argument("-col", "--collect", action="store_true", help="If given, run the data collection process")
procedure.add_argument("-prep", "--preprocess", action="store_true", help="If given, showcase various preprocessing pipelines")
procedure.add_argument("-tr", "--train", action="store_true", help="If given, run the model training process")
procedure.add_argument("-show", "--showcase", action="store_true", help="If given, run the showcasing process")
procedure.add_argument("-pred", "--predict", action="store_true", help="If given, run the showcasing process with model prediction")

# Specify training settings
train_settings = parser.add_argument_group("Training settings")
train_settings.add_argument("--experiment", default=None, type=int,
                            help="Number of this experiment (the settings will be saved in the respective newly created folder or loaded from an existing folder)")
train_settings.add_argument("-tb", "--tensorboard", action="store_true", help="If given, set up TensorBoard callback for model training")
train_settings.add_argument("-es", "--early_stopping", default="loss", choices=["disable", "loss", "accuracy", "f1_score", "recall", "precision"], help="Choice of the metric to monitor by the EarlyStopping callback during model training")
train_settings.add_argument("-aug", "--augmentation", action="store_true", help="If given, perform image augmentation on the training dataset")
# TODO - Decide if this is a good approach # train_settings.add_argument("-efnet", "--efficient_net", action="store_true", help="If given, omit training of a new model and only finetune the output layers of the EfficientNetV2B0")

# Specify the hyperparameters if the json file was not given
hyperparameters = parser.add_argument_group("Hyperparameters")
hyperparameters.add_argument("-bs", "--batch_size", default=128, type=int, help="Batch size")
hyperparameters.add_argument("-e", "--epochs", default=10, type=int, help="Number of epochs")
hyperparameters.add_argument("-opt", "--optimizer", default="adam", choices=["adam", "SGD"], help="Optimizer for training")
hyperparameters.add_argument("-lr", "--learning_rate", default=0.01, type=float, help="Starting learning rate")
hyperparameters.add_argument("-lrd", "--lr_decay", default=None, type=float, help="If given, this constant will be used for exponential learning rate decay after every epoch")
hyperparameters.add_argument("-mom", "--momentum", default=None, type=float, help="If optimizer is set to SGD, initialize the optimizer with Nesterov momentum of this value if given")
hyperparameters.add_argument("-wd", "--weight_decay", default=None, type=float, help="If given, set the weight decay for the optimizer to this value")
hyperparameters.add_argument("--seed", default=123, type=int, help="Random seed for operations including randomness (e.g. shuffling)")
hyperparameters.add_argument("--split", default=0.3, type=float, help="Portion of the full dataset to reserve for validation")

# Specify the architecture for the given experiment if training procedure is set
architecture = parser.add_argument_group("Architecture")
architecture.add_argument("-arch", "--architecture", default=None, type=str,
                          help="Specify the trainable layers for the network using the build_model function (for more information, see its documentation)")
architecture.add_argument("-prep_layers", "--preprocessing_layers", default=None, type=str,
                          help="Specify the architecture of the preprocessing pipeline using the build_preprocessing function (for more information, see its documentation)")


def main(args):
    """Command line run function"""

    # Optically indent the beginning of the process
    utils.indent()

    # Load configuration file from json in the given folder
    with open(args.config_dir + "config.json", "r") as config_file:
        config = json.load(config_file)

    # Default procedure
    if not (args.collect or args.preprocess or args.train or args.showcase or args.predict):
        print("\nNo arguments specified, will try to run the showcasing without prediction\n")
        args.showcase = True

    # Set up the list of gestures
    with open(config["Paths"]["Gesture list"], "r") as gesture_list:
        gestures = gesture_list.readlines()[0].split(", ")

    # Set up folders and necessary variables
    script_dir = os.path.dirname("run.py")
    data_dir, example_dir, model_dir, desired_amount, current_amount, paths = utils.setup_folders(script_directory=script_dir,
                                                                                                  gestures_list=gestures,
                                                                                                  amount_per_gesture=config["General parameters"]["Desired amount"])
    experiments_dir = os.path.join(model_dir, "experiments")
    current_dir = os.path.join(model_dir, "current")
    img_size = config["General parameters"]["Image size"]
    print("The folders have been set up.")
    utils.indent()

    # Collection of the data
    if args.collect:

        print("Starting the data collection process.")

        # If the run.py is executed with desired amount already filled, the
        # user has an option to top-up the desired amount by a bit
        if all(cur >= des for des, cur in zip(desired_amount.values(),
                                              current_amount.values())):
            increment = config["General parameters"]["Top-up amount"]
            print("You are trying to run the collection procedure even though",
                  "the desired amount has already been reached.",
                  "Do you want to increase the current amount per gesture",
                  f"by {increment}?")
            if input("Proceed (y/[n])?").lower() == "y":
                desired_amount = {letter: current_amount[letter] + increment for letter in current_amount.keys()}

        # Perform the dataset collection procedure
        collect_data(gestures, examples=example_dir,
                     data_directory=data_dir, current_amounts=current_amount,
                     desired_amounts=desired_amount, gesture_paths=paths,
                     translations=config["Paths"]["Translations"],
                     img_size=img_size)
        print("Your data has been collected, please check the folders.")

    # Showcase various preprocessing pipelines and enable saving their outputs
    elif args.preprocess:

        print("Starting to showcase different preprocessing pipelines")

        showcase_preprocessing(inp_shape=[img_size, img_size, 3])

    # Build a new model and train it on the given data
    elif args.train:

        print("Starting the model training process")

        # Set threading options to autotuning
        tf.config.threading.set_inter_op_parallelism_threads(0)
        tf.config.threading.set_intra_op_parallelism_threads(0)

        # Cover the case where the user wishes to save the model as current
        if args.experiment == -1:

            # Ask for confirmation in case the current folder already has some saved model in it
            if len(list(os.walk(current_dir))):
                print("The folder with current model layout already contains some files.",
                      "Continuing to save the result of the current training procedure",
                      "might result in loss of the previous model.")
                if input("Proceed (y/[n])?").lower() != "y":
                    print("Aborting the training procedure.")
                    return

            # Save the model as the current version
            save_dir = current_dir

        else:
            # Adjust the experiment number accordingly if not given
            if args.experiment is None:
                exp_folders = list(next(os.walk(experiments_dir))[1])
                if not exp_folders:
                    args.experiment = 1
                else:
                    exp_folders.sort(key=lambda folder: int(re.split(r"[_]", folder)[1]))
                    args.experiment = int(re.search(r"\d+", exp_folders[-1]).group()) + 1
                print("You selected training procedure but did not input an experiment number.",
                      f"A new folder with experiment number {args.experiment} will thus be created for this experiment.")

            experiment_dir = os.path.join(experiments_dir, "experiment_" + str(args.experiment))

            # Ask for confirmation if the given experiment folder already exists and create new one if prompted
            if os.path.exists(experiment_dir):
                print(f"A folder for the given experiment number ({args.experiment}) already exists.")
                print("Saving the train procedure for this run in the given folder might result in",
                      "overwriting of previously saved models.")
                if input("Proceed (y/[n])?").lower() != "y":
                    if input("Do you wish to automatically create a new folder for this run and then continue (y/[n])?").lower() == "y":
                        exp_folders = list(next(os.walk(experiments_dir))[1])
                        exp_folders.sort(key=lambda folder: int(re.split(r"[_]", folder)[1]))
                        args.experiment = int(re.search(r"\d+", exp_folders[-1]).group()) + 1
                        experiment_dir = os.path.join(experiments_dir, "experiment_" + str(args.experiment))
                        print(f"A new folder with experiment number {args.experiment} will be created for this experiment.")
                    else:
                        print("Aborting the training procedure.")
                        return

            # Save the model in the respective experiment folder
            utils.new_folder(experiment_dir)
            save_dir = experiment_dir

        # Optically indent the preparation from actual model building
        utils.indent()

        # Loading the training and testing datasets from directories and optimizing them for performance
        train_images, test_images = tf.keras.preprocessing.image_dataset_from_directory(data_dir,
                                                                                        labels="inferred",
                                                                                        label_mode="categorical",
                                                                                        class_names=gestures,
                                                                                        color_mode="rgb",
                                                                                        batch_size=args.batch_size,
                                                                                        image_size=(img_size,
                                                                                                    img_size),
                                                                                        shuffle=True,
                                                                                        seed=args.seed,
                                                                                        validation_split=args.split,
                                                                                        subset="both")

        if args.augmentation:
            augmentation_model = image_augmentation(seed=args.seed)
            train_images = train_images.map(lambda x, y: (augmentation_model(x, training=True), y),
                                            num_parallel_calls=tf.data.AUTOTUNE).prefetch(buffer_size=tf.data.AUTOTUNE)
        else:
            train_images = train_images.prefetch(buffer_size=tf.data.AUTOTUNE)
        test_images = test_images.prefetch(buffer_size=tf.data.AUTOTUNE)

        # Set up the log directories for checkpoints and tensorboard
        cp_path = os.path.join(save_dir, "ckpt/cp-{epoch:03d}.ckpt")
        tb_path = os.path.join(config["Paths"]["Logs"],
                               "{}".format(datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")))

        # Create callbacks for the model to save progress during training
        # Checkpoint callback
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=cp_path,
                                                         verbose=1,
                                                         save_weights_only=True,
                                                         save_freq="epoch")
        callbacks = [cp_callback]

        # TensorBoard callback (optional, default is to not include)
        if args.tensorboard:
            tb_callback = tf.keras.callbacks.TensorBoard(log_dir=tb_path,
                                                         histogram_freq=1)
            callbacks.append(tb_callback)

            # Add functionality to save log confusion matrices
            writer = tf.summary.create_file_writer(logdir=tb_path + "/cm")
            cm_callback = ConfusionMatrixCallback(writer,
                                                  gesture_list=gestures,
                                                  validation_data=test_images)
            callbacks.append(cm_callback)

        # Early stopping callback (optional, default is to include)
        if args.early_stopping != "disable":
            es_callback = tf.keras.callbacks.EarlyStopping(monitor="val_" + args.early_stopping,
                                                           min_delta=0.001,
                                                           patience=5,
                                                           verbose=1,
                                                           mode="auto",
                                                           baseline=0.3,
                                                           restore_best_weights=True,
                                                           start_from_epoch=10)
            callbacks.append(es_callback)

        # Set the default preprocessing pipeline if not specified
        if args.preprocessing_layers is None:
            args.preprocessing_layers = config["Model"]["Default preprocessing"]

        # Build the preprocessing pipeline according to given instructions
        preprocessing = build_preprocessing(inp_shape=[img_size,
                                                       img_size,
                                                       3],
                                            instructions=args.preprocessing_layers,
                                            name="preprocessing_pipeline")

        # Set the default model architecture if not specified
        if args.architecture is None:
            args.architecture = config["Model"]["Default architecture"]

        # Adjust the number of input channels for the trainable layers based on grayscale layer presence
        channels = 1 if "G" in args.preprocessing_layers else 3

        # Build the model according to given instructions
        trainable = build_model(inp_shape=[img_size,
                                           img_size,
                                           channels],
                                output_size=len(gestures),
                                instructions=args.architecture,
                                name="trainable_layers")

        # Merge the preprocessing pipeline with the trainable layers
        model = tf.keras.Model(inputs=preprocessing.input,
                               outputs=trainable(preprocessing.output),
                               name="full_model")

        # Compile the modile according to given instructions
        if args.optimizer == "adam":
            optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate,
                                                 weight_decay=args.weight_decay,
                                                 jit_compile=False)
        elif args.optimizer == "SGD":
            optimizer = tf.keras.optimizers.experimental.SGD(learning_rate=args.learning_rate,
                                                             nesterov=True,
                                                             momentum=args.momentum,
                                                             weight_decay=args.weight_decay,
                                                             jit_compile=False)
        model.compile(optimizer=optimizer,
                      loss=tf.keras.losses.CategoricalCrossentropy(),
                      metrics=[tf.keras.metrics.CategoricalAccuracy(name="accuracy"),
                               tf.keras.metrics.Recall(name="recall"),
                               tf.keras.metrics.Precision(name="precision"),
                               tfa.metrics.F1Score(len(gestures),
                                                   average="macro",
                                                   name="f1_score")])

        # Show summaries for all the models
        utils.indent(n=2)
        print("Models have been built, showing model summaries now.\n")
        print("Preprocessing pipeline:")
        print(preprocessing.summary())
        utils.indent(n=1)
        print("Trainable layers of the model:")
        print(trainable.summary())
        utils.indent(n=1)
        print("Fully compiled model:")
        print(model.summary())
        utils.indent(n=2)
        print("Model training:")
        # Save the initial weights as specified in the "checkpoint_path" format
        model.save_weights(cp_path.format(epoch=0))

        # Train the model according to the given instructions
        history = model.fit(train_images, validation_data=(test_images),
                            epochs=args.epochs,
                            callbacks=callbacks,
                            verbose=1)

        # Save the model into the appropriate folder
        model.save(filepath=save_dir,
                   overwrite=True)

        # Save the model architecture in a text file as well for easy access
        with open(os.path.join(save_dir, "model_summary.txt"), "a+") as file:
            file.write("Preprocessing pipeline:\n")
            preprocessing.summary(print_fn=lambda x: file.write(x + "\n"))
            file.write("\n\n")
            file.write("Trainable summary:\n")
            trainable.summary(print_fn=lambda x: file.write(x + "\n"))
            file.write("\n\n")
            file.write("\nTraining parameters: " + str(history.params) + "\n")
            file.write("Final training accuracy: " + str(history.history["accuracy"][-1]) + "\n")
            file.write("Final validation accuracy: " + str(history.history["val_accuracy"][-1]) + "\n")
            file.write("Final training precision: " + str(history.history["precision"][-1]) + "\n")
            file.write("Final validation precision: " + str(history.history["val_precision"][-1]) + "\n")
            file.write("Final training recall: " + str(history.history["recall"][-1]) + "\n")
            file.write("Final validation recall: " + str(history.history["val_recall"][-1]) + "\n")
            file.write("Final training f1_score: " + "\n" + str(history.history["f1_score"][-1]) + "\n")
            file.write("Final validation f1_score: " + "\n" + str(history.history["val_f1_score"][-1]) + "\n")
            file.write("\n\n")
            file.write("Command line arguments: " + "\n")
            for arg in vars(args):
                file.write(arg + ": " + str(getattr(args, arg)) + "\n")

    # Demonstrate the image taking process
    elif args.showcase:

        print("Starting environment showcasing process")

        showcase_model(gestures, examples=example_dir,
                       predict=False, model=None,
                       translations=config["Paths"]["Translations"],
                       img_size=img_size)

    # Demonstrate the image taking process while also demonstrating the model and its predictions
    elif args.predict:

        print("Starting model demonstration process")

        try:
            model = tf.keras.models.load_model(filepath=config["Model"]["Current model"],
                                               custom_objects={"AdaptiveThresholding": AdaptiveThresholding,
                                                               "Blurring": Blurring,
                                                               "Grayscale": Grayscale})
        except IOError:
            print("The prediction procedure was chosen but model cannot be found",
                  f"in the folder specified in the config.json file ({config['Model']['Current model']}).\n",
                  "Please make sure to adjust the folder name in the config file or save the model in there.\n",
                  "Eventually, this can be resolved by running the train procedure with experiment number specified as -1.\n")
            print("Terminating the prediction process.")
            return

        showcase_model(gestures, examples=example_dir,
                       predict=True, model=model,
                       translations=config["Paths"]["Translations"],
                       img_size=img_size)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
