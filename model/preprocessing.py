"""
Image preprocessing functionalities for the Bachelor's Thesis project
on the following topic:
    "Construction of a Neural Networks model for translation of recorded sign language".
Modularized and parametrized preprocessing pipeline utilities for RGB images.

@author: Tomáš Děd
"""

import cv2
import tensorflow as tf
import tensorflow_addons as tfa


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

    def call(self, input_batch):
        """
        Actions to perform on the input batch during layer call within tf.keras model

        Parameters
            input_batch: tf.data.dataset
                Batch for the layer to perform operations on

        Returns
            tf.data.dataset
                The dataset on which the apply_threshold function was mapped (applied to each element)

        """

        def apply_thresholding(image):
            """
            Apply adaptive thresholding on the given image (tf.Tensor object)

            Parameters
                image: tf.Tensor (dtype tf.uint8)
                    Image to apply adaptive thresholding on.

            Returns
                np.ndarray
                    Numpy array that corresponds to the thresholded image
            """

            # OpenCV adaptive thresholding only accepts numpy arrays
            img = image.numpy()

            # Perform thresholding based on specified type
            if self.thresholding_type == "mean":
                thresholded_img = cv2.adaptiveThreshold(img, 255,
                                                        cv2.ADAPTIVE_THRESH_MEAN_C,
                                                        cv2.THRESH_BINARY_INV,
                                                        self.block_size, self.constant)

            else:
                thresholded_img = cv2.adaptiveThreshold(img, 255,
                                                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                        cv2.THRESH_BINARY_INV,
                                                        self.block_size, self.constant)

            # Output for this function should be tensors of shape (img_size, img_size, 1)
            return tf.expand_dims(tf.convert_to_tensor(thresholded_img,
                                                       dtype=tf.uint8),
                                  axis=-1)

        # Perform the thresholding per image in a batch
        return tf.map_fn(lambda image: tf.py_function(func=apply_thresholding,
                                                      inp=[image],
                                                      Tout=tf.uint8),
                         input_batch)

    def get_config(self):
        """
        Return configuration of the layer for the purpose of model saving
        """

        config = super(AdaptiveThresholding, self).get_config()
        config.update({"thresholding_type": self.thresholding_type,
                       "block_size": self.block_size,
                       "constant": self.constant})

        return config


class Blurring(tf.keras.layers.Layer):
    """
    A class for tensorflow.keras Blurring layer

    Methods:
        call(input_batch)
            Operations performed on layer call within a tensorflow.keras model
        get_config()
            Output configuration for the layer for the purpose of model saving
    """

    def __init__(self, blurring_type="median", kernel_size=3, sigma=1, **kwargs):
        """

        Parameters:
            blurring_type: str (default "median", optional)
                What type of blurring to perform (options are median and gaussian).
                Gaussian thresholding is selected for strings other than "median"
            kernel_size: positive int (default 3, optional)
                The size of the kernel within blurring function
            sigma: positive float (default 1, optional)
                The sigma parameter for gaussian blurring (irrelevant for median blur)
            **kwargs:
                Keyword arguments inherited from the tf.keras.layers.Layer class
        """

        super(Blurring, self).__init__(**kwargs)
        self.blurring_type = blurring_type
        self.kernel_size = kernel_size
        self.sigma = sigma

    def call(self, input_batch):
        """
        Actions to perform on the input batch during layer call within tf.keras model

        Parameters
            input_batch: tf.data.dataset
                Batch for the layer to perform operations on

        Returns
            tf.data.dataset
                The dataset on which the tfa.image filter function was applied
        """

        # Perform blurring based on specified type
        if self.blurring_type == "median":
            return tfa.image.median_filter2d(input_batch, self.kernel_size)

        return tfa.image.gaussian_filter2d(input_batch, self.kernel_size, self.sigma)

    def get_config(self):
        """
        Return configuration of the layer for the purpose of model saving
        """

        config = super(Blurring, self).get_config()
        config.update({"blurring_type": self.blurring_type,
                       "kernel_size": self.kernel_size,
                       "sigma": self.sigma})

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

    def call(self, input_batch):
        """
        Actions to perform on the input batch during layer call within tf.keras model

        Parameters
            input_batch: tf.data.dataset
                Batch for the layer to perform operations on

        Returns
            tf.data.dataset
                The dataset on which the tfa.image.rgb_to_grayscale function was applied
        """

        return tf.image.rgb_to_grayscale(input_batch)

    def get_config(self):
        """
        Return configuration of the layer for the purpose of model saving
        """

        return super(Grayscale, self).get_config()
