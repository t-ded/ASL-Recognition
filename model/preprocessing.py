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

    def __init__(self, thresholding_type="mean", block_size=3, constant=0, **kwargs):

        super(AdaptiveThresholding, self).__init__(**kwargs)
        self.thresholding_type = thresholding_type
        self.block_size = block_size
        self.constant = constant

    def call(self, input_batch):

        def apply_thresholding(image):

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

        config = super(AdaptiveThresholding, self).get_config()
        config.update({"thresholding_type": self.thresholding_type,
                       "block_size": self.block_size,
                       "constant": self.constant})

        return config


class Blurring(tf.keras.layers.Layer):

    def __init__(self, blurring_type="median", kernel_size=3, sigma=1, **kwargs):

        super(Blurring, self).__init__(**kwargs)
        self.blurring_type = blurring_type
        self.kernel_size = kernel_size
        self.sigma = sigma

    def call(self, input_batch):

        # Perform blurring based on specified type
        if self.blurring_type == "median":
            return tfa.image.median_filter2d(input_batch, self.kernel_size)

        return tfa.image.gaussian_filter2d(input_batch, self.kernel_size, self.sigma)

    def get_config(self):

        config = super(Blurring, self).get_config()
        config.update({"blurring_type": self.blurring_type,
                       "kernel_size": self.kernel_size,
                       "sigma": self.sigma})

        return config


class Grayscale(tf.keras.layers.Layer):

    def __init__(self):

        super(Grayscale, self).__init__()

    def call(self, input_batch):

        return tf.image.rgb_to_grayscale(input_batch)

    def get_config(self):

        return super(Grayscale, self).get_config()
