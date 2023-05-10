"""
Image preprocessing functionalities for the Bachelor's Thesis project
on the following topic:
    "Construction of a Neural Networks model for translation of recorded sign language".
Modularized and parametrized preprocessing pipeline utilities for RGB images.

@author: Tomáš Děd
"""

import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import matplotlib.pyplot as plt
import io
from itertools import product
from sklearn.metrics import confusion_matrix


class AdaptiveThresholding(tf.keras.layers.Layer):
    """
    A class for tensorflow.keras AdaptiveThresholding layer

    Methods:
        call(input_batch)
            Operations performed on layer call within a tensorflow.keras model
        get_config()
            Output configuration for the layer for the purpose of model saving
    """

    def __init__(self, thresholding_type="mean", block_size=3, constant=0, **kwargs):
        """

        Parameters:
            thresholding_type: str (default "mean", optional)
                What type of adaptive thresholding to perform (options are mean and gaussian).
                Gaussian thresholding is selected for strings other than "mean"
            block_size: positive int (default 3, optional)
                The size of the block within thresholding function
            constant: float (default 0, optional)
                The constant to add within adaptive thresholding
            **kwargs:
                Keyword arguments inherited from the tf.keras.layers.Layer class
        """

        super(AdaptiveThresholding, self).__init__(**kwargs)
        self.thresholding_type = thresholding_type
        self.block_size = block_size
        self.constant = constant
        self.trainable = False
        if self.thresholding_type == "mean":
            self.kernel = tf.ones([self.block_size,
                                   self.block_size,
                                   1,
                                   1], dtype="float32") / (self.block_size ** 2)

    def call(self, input_batch):
        """
        Actions to perform on the input batch during layer call within tf.keras model

        Parameters:
            input_batch: tf.data.dataset
                Batch for the layer to perform operations on

        Returns:
            output_batch
                The batch with each image thresholded using given thresholding type

        """

        if self.thresholding_type == "mean":

            # Obtain mean of each block per batch sample
            # via a convolution with an appropriate kernel
            mean = tf.nn.conv2d(input_batch,
                                filters=self.kernel,
                                strides=1,
                                padding="SAME")

            # Mask and threshold the input batch by the means
            # Transpose for effective NHWC <-> NCHW conversion during TensorFlow's training
            return tf.transpose(tf.where(tf.transpose(input_batch <= mean - self.constant,
                                                      perm=[0, 3, 1, 2]),
                                         255, 0),
                                perm=[0, 2, 3, 1])

        else:

            # Obtain gaussian weighted sum of each block per batch sample
            # via a convolution with an appropriate kernel
            gaussian_weighted = tfa.image.gaussian_filter2d(input_batch,
                                                            filter_shape=self.block_size)

            # Mask and threshold the input batch by the gaussian weighted sums
            # Transpose for effective NHWC <-> NCHW conversion during TensorFlow's training
            return tf.transpose(tf.where(tf.transpose(input_batch <= gaussian_weighted - self.constant,
                                                      perm=[0, 3, 1, 2]),
                                         255, 0),
                                perm=[0, 2, 3, 1])

    def compute_output_shape(self, input_shape):
        """
        Compute the output shape of this layer based on the given input shape

        Parameters:
            input_shape: tuple or tf.TensorShape
                Specification of the input shape for this layer
        Returns:
            tuple of 4 ints
                Output shape (same as the input shape)
        """

        return tf.TensorShape([*input_shape])

    def get_config(self):
        """
        Return configuration of the layer for the purpose of model saving
        """

        config = super(AdaptiveThresholding, self).get_config()
        config.update({"thresholding_type": self.thresholding_type,
                       "block_size": self.block_size,
                       "constant": self.constant})

        return config


class Grayscale(tf.keras.layers.Layer):
    """
    A class for tensorflow.keras Grayscale layer

    Methods:
        call(input_batch)
            Operations performed on layer call within a tensorflow.keras model
        get_config()
            Output configuration for the layer for the purpose of model saving
    """

    def __init__(self, **kwargs):
        """

        Parameters:
            **kwargs:
                Keyword arguments inherited from the tf.keras.layers.Layer class
        """

        super(Grayscale, self).__init__()
        self.trainable = False

    def call(self, input_batch):
        """
        Actions to perform on the input batch during layer call within tf.keras model

        Parameters:
            input_batch: tf.data.dataset
                Batch for the layer to perform operations on

        Returns:
            output_batch
                The batch on which the tfa.image.rgb_to_grayscale function was mapped
        """

        return tf.image.rgb_to_grayscale(input_batch)

    def get_config(self):
        """
        Return configuration of the layer for the purpose of model saving
        """

        return super(Grayscale, self).get_config()


class LinearWarmupCallback(tf.keras.callbacks.Callback):
    """
    A callback to linearly, gradually ramp up the learning rate during
    early stages of training

    Methods:
        adjust_lr(global_step, warmup_step, total_steps, start_lr, target_lr)
            Determine the value for the learning rate at given point in training
        on_training_batch_end(batch, logs)
            Operations performed at the end of every batch with the result possibly saved to the logdir
        on_training_batch_start(batch, logs)
            Operations performed at the start of every batch with the result possibly saved to the logdir
    """

    def __init__(self, warmup_steps=0,
                 start_lr=0.0, target_lr=0.01, **kwargs):
        """

        Parameters:
            warmup_steps: int (default 0)
                Number of steps in which to go from start_lr to target_lr
            start_lr: float (default 0.0)
                Learning rate to use as the initial one
            target_lr: float (default 0.01)
                Learning rate to which to get in warmup_steps from start_lr
            **kwargs:
                Keyword arguments inherited from the tf.keras.callbacks.Callback class
        """

        super(LinearWarmupCallback, self).__init__()
        self.warmup_steps = warmup_steps
        self.start_lr = start_lr
        self.target_lr = target_lr
        self.global_step = 0

        # Set up the list of learning rates for the early steps
        if self.warmup_steps <= 0:
            self.lrs = [target_lr]
        else:
            self.lrs = tf.linspace(self.start_lr, self.target_lr, self.warmup_steps + 1)

    def on_train_batch_end(self, batch, logs=None):
        """Operations to perform at the beginning of the given batch"""

        self.global_step = self.global_step + 1

    def on_train_batch_begin(self, batch, logs=None):
        """Operations to perform at the end of the given batch"""

        # Update the learning rate in early stages only
        if self.global_step <= self.warmup_steps:
            lr = self.lrs[self.global_step]
            tf.keras.backend.set_value(self.model.optimizer.lr, lr)


class LRTensorBoardCallback(tf.keras.callbacks.TensorBoard):
    """
    A callback to log the learning rate to TensorBoard

    Methods:
        on_epoch_end(epoch, logs)
            Operations performed at the end of every epoch with the result possibly saved to the logdir
    """

    def __init__(self, logs=None, **kwargs):
        """

        Parameters:
            logs: str (default None)
                If given, log the learning rate to the logs directory to visualize in TensorBoard
            **kwargs:
                Keyword arguments inherited from the tf.keras.callbacks.Callback class
        """

        super().__init__(log_dir=logs, **kwargs)

    def on_epoch_end(self, epoch, logs=None):
        """Operations to perform at the end of the given epoch"""

        logs = logs or {}
        logs.update({"lr": tf.keras.backend.eval(self.model.optimizer.lr)})
        super().on_epoch_end(epoch, logs)


class ConfusionMatrixCallback(tf.keras.callbacks.Callback):
    """
    A class to save confusion matrix for model's prediction at the end of every epoch

    Methods:
        plot_confusion_matrix(cm)
            Return figure of the given confusion matrix
        on_epoch_end(epoch, logs)
            Operations performed at the end of every epoch with the result saved to the logdir
    """

    def __init__(self, writer, gesture_list,
                 validation_data=None, **kwargs):
        """

        Parameters:
            writer: tf.SummaryWriter
                TensorFlow summary writer with the designated log dir for this callback
            gesture_list: list of str (default None)
                List of gestures to put on the axes
            validation_data: tf.data.Dataset (default None)
                Validation dataset containing (sample, label) pairs
            **kwargs:
                Keyword arguments inherited from the tf.keras.callbacks.Callback class
        """

        super(ConfusionMatrixCallback, self).__init__()
        self.gesture_list = gesture_list
        self.validation_data = validation_data
        self.writer = writer

    def plot_confusion_matrix(self, cm):
        """
        Create figure for the given confusion matrix

        Parameters:
            cm: numpy.ndarray
                The given confusion matrix to plot

        Returns:
            figure: matplotlib.pyplot figure
        """

        # Set up the figure
        figure = plt.figure(figsize=(28, 28))
        plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Reds)
        plt.colorbar(shrink=0.75).ax.tick_params(labelsize=18)

        # Set up axes
        tick_marks = np.arange(len(np.unique(self.gesture_list)))
        plt.xticks(tick_marks, self.gesture_list, rotation=90, fontsize=20)
        plt.yticks(tick_marks, self.gesture_list, fontsize=20)

        # White text for darker squares, otherwise black text
        thresh = cm.max() / 2.
        for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, f"{cm[i, j]:.2f}",
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        # Finalize the figure
        plt.ylabel("True label", fontsize=30)
        plt.xlabel("Predicted label", fontsize=30)
        plt.tight_layout()

        return figure

    def on_epoch_end(self, epoch, logs=None):
        """Operations to perform at the end of the given epoch"""

        # Compute the confusion matrix per batch
        y_val = np.concatenate([np.argmax(y, axis=1) for x, y in self.validation_data], axis=0)
        y_pred = np.argmax(self.model.predict(self.validation_data), axis=1)
        cm = confusion_matrix(y_val, y_pred,
                              normalize="true")

        # Create the figure using previously created function
        figure = self.plot_confusion_matrix(cm)

        # Save confusion matrix to TensorBoard log directory
        with self.writer.as_default():
            buf = io.BytesIO()
            plt.savefig(buf, format="png")
            plt.close(figure)
            buf.seek(0)
            image = tf.image.decode_png(buf.getvalue(), channels=4)
            image = tf.expand_dims(image, 0)
            tf.summary.image("epoch_confusion_matrix", image, step=epoch)
            plt.close("all")

        buf.close()


def image_augmentation(seed=123, rot_factor=0.03,
                       height_factor=0.05, width_factor=0.05):
    """
    Build a data augmentation tf.keras.Sequential model

    Parameters:
        seed: int (default 123)
            Seed for the random operations
        rot_factor: float (default 0.05)
            Rotation factor (as a fraction of 2 Pi) to use for rotation bounds
        height_factor: float (default 0.1)
            Factor which to use for height shift bounds
        width_factor: float (default 0.1)
            Factor which to use for width shift bounds

    Returns
        tf.keras.Sequential
            Sequential model with augmentation layers
    """

    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip(mode="horizontal",
                                       seed=seed),
            tf.keras.layers.RandomRotation(factor=rot_factor,
                                           fill_mode="nearest",
                                           seed=seed),
            tf.keras.layers.RandomTranslation(height_factor=height_factor,
                                              width_factor=width_factor,
                                              fill_mode="nearest",
                                              seed=seed)
        ],
        name="image_augmentation"
    )

    return augmentation
