import tensorflow as tf
import numpy as np


def weight_variable_glorot(input_dim, output_dim, name=""):
    """Create a weight variable with Glorot & Bengio (AISTATS 2010)
    initialization.
    """
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    initial = tf.random.uniform([input_dim, output_dim], minval=-init_range,
                                maxval=init_range, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def zeros(input_dim, output_dim, name=None):
    """All zeros."""
    initial = tf.zeros((input_dim, output_dim), dtype=tf.float32)
    return tf.Variable(initial, name=name)


def ones(input_dim, output_dim, name=None):
    """All zeros."""
    initial = tf.ones((input_dim, output_dim), dtype=tf.float32)
    return tf.Variable(initial, name=name)