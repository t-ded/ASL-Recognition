"""
Image preprocessing functionalities for the Bachelor's Thesis project
on the following topic:
    "Construction of a Neural Networks model for translation of recorded sign language".
Modularized and parametrized preprocessing pipeline utilities for RGB images.

@author: Tomáš Děd
"""

import cv2
import tensorflow as tf


class AdaptiveThresholding(tf.keras.layers.Layer):

    def __init__(self, thresholding_type="mean", block_size=3, constant=0, **kwargs):

        super(AdaptiveThresholding, self).__init__(**kwargs)
        self.thresholding_type = thresholding_type
        self.block_size = block_size
        self.constant = constant

    def call(self, input_batch):

        def apply_thresholding(image):

            if self.thresholding_type == "mean":
                return cv2.adaptiveThresholding(image, 1,
                                                cv2.ADAPTIVE_THRESH_MEAN_C,
                                                cv2.THRESH_BINARY_INV,
                                                self.block_size, self.constant)

            return cv2.adaptiveThresholding(image, 1,
                                            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY_INV,
                                            self.block_size, self.constant)

        return tf.py_function(func=apply_thresholding, inp=[input_batch], Tout=tf.float32)


class Blurring(tf.keras.layers.Layer):

    def __init__(self, blurring_type="median", kernel_size=3, sigma=None, **kwargs):

        super(Blurring, self).__init__(**kwargs)
        self.blurring_type = blurring_type
        self.kernel_size = kernel_size
        self.sigma = sigma

    def call(self, input_batch):

        def apply_blurring(image):

            if self.blurring_type == "median":
                return cv2.medianBlur(image, self.kernel_size)

            return cv2.gaussianBlur(image, (self.kernel_size, self.kernel_size),
                                    sigma_X=self.sigma, sigma_Y=self.sigma)

        return tf.py_function(func=apply_blurring, inp=[input_batch], Tout=tf.float32)
