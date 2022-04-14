""" Script for generating data. Thompson 2022"""
import argparse
from constants import GENERATOR_NAME, LATENT_DIM, SINGLE_LABELS, MULTI_LABELS, LABEL_CHOICES
import numpy as np
import numpy.random as np_rand
import tensorflow as tf
import os.path
from tensorflow import keras
from tensorflow.keras.models import load_model


def random_labels(num_signals: int, label_type: int = 0):
    """ Generates and returns number of random labels.
    :param num_signals: int
    :param label_type: int - Can be 0 for single labels, 1 for multilabels
    , or 2 for both.
    :returns: np.ndarray - array of labels of length equal to num_signals."""
    if num_signals <= 0:
        raise ValueError(f'num_signals must be positive, {num_signals} not allowed.')
    label_arr = None
    if label_type == 0:
        label_arr = SINGLE_LABELS
    elif label_type == 1:
        label_arr = MULTI_LABELS
    elif label_type == 2:
        label_arr = LABEL_CHOICES
    else:
        raise ValueError(f'label_type must be 0, 1, or 2. {label_type} is not a valid input.')
    rng = np.random.default_rng()
    labels = rng.choice(label_arr, size=(num_signals,))  # Choose labels from pool
    return labels


def generate_data(num_signals: int, generator: str, labels=None):
    """ """
    # if num_signals <= 0:
    #     raise ValueError(f'num_signals must be positive, {num_signals} not allowed.')
    if not (labels is None or isinstance(labels, (int, list, tuple, np.ndarray))):
        raise TypeError('labels must be int, list, or tuple')  # Might make better later
    model = load_model(generator)
    fn_labels = None
    if labels is None:
        fn_labels = random_labels(num_signals)
    elif isinstance(labels, int):
        fn_labels = random_labels(num_signals, labels)
    elif isinstance(labels, (list, tuple, np.ndarray)):
        if len(labels) == 1:
            fn_labels = np.repeat(labels, num_signals, axis=0)
        elif len(labels) == num_signals:
            fn_labels = np.asarray(labels)
        else:
            # Unsure if this will allow choosing from selection, tiled repeat, or error
            raise NotImplementedError
    return model.predict(
        (tf.random.normal(shape=(num_signals, LATENT_DIM)), fn_labels))


def save_array(arr, filename: str, overwrite=False):
    """ Save array of generated signals.
        :param: arr -
        :param: filename - str
        :return: None"""
    if not filename.endswith(('.npy', '.txt', '.gz')):
        raise NotImplementedError
    if not overwrite and os.path.exists(filename):
        raise FileExistsError(
            f'Writing to path would overwrite file. Path {filename}')
    if filename.endswith('.npy'):
        np.save(filename, arr)
    elif filename.endswith('.txt') or filename.endswith('.gz'):
        np.savetxt(filename, arr)


def parser():
    """ Command line parser for program."""
    pass


def main():
    # TODO make some number of generated signals
    # TODO should be able to specify label(s)
    # TODO save options should be .npy or .txt
    pass


if __name__ == '__main__':
    main()
