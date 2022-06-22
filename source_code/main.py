""" Script for generating data. Thompson 2022"""
import argparse
import os
import os.path

import constants
from constants import GENERATOR_NAME, LATENT_DIM, SINGLE_LABELS,\
    MULTI_LABELS, LABEL_CHOICES

import numpy as np
import numpy.random as np_rand
# Must come before tensorflow import to suppress warnings
import warning_suppressor  # Calls line below while keeping imports valid
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

from tensorflow.keras.models import load_model


def random_labels(num_signals: int, label_type: int = 0):
    """ Generates and returns number of random labels.

    :param num_signals: int
    :param label_type: int - Can be 0 for single labels, 1 for multilabels, or 2 for both.
    :returns: np.ndarray - array of labels of length equal to num_signals."""
    if num_signals <= 0:
        raise ValueError(
            f'num_signals must be positive, {num_signals} not allowed.')
    label_arr = None
    if label_type == 0:
        label_arr = SINGLE_LABELS
    elif label_type == 1:
        label_arr = MULTI_LABELS
    elif label_type == 2:
        label_arr = LABEL_CHOICES
    else:
        raise ValueError(
            f'label_type must be 0, 1, or 2. {label_type} is not a valid input.')
    rng = np.random.default_rng()
    labels = rng.choice(label_arr, size=(num_signals,))  # Choose labels from pool
    return labels


def generate_data(num_signals: int, generator: str, labels=None):
    """ Function to generate data.
        :param num_signals: int
        :param generator: str
        :param labels: int, list, tuple, ndarray, None"""
    # if num_signals <= 0:
    #     raise ValueError(f'num_signals must be positive, {num_signals} not allowed.')
    if not (labels is None or isinstance(labels, (int, list, tuple, np.ndarray))):
        raise TypeError('labels must be int, list, or tuple')  # Might make better later
    model = load_model(generator, compile=False)
    fn_labels = None
    if labels is None:
        fn_labels = random_labels(num_signals)
    elif isinstance(labels, int):
        fn_labels = random_labels(num_signals, labels)
    elif isinstance(labels, (list, tuple, np.ndarray)):
        if len(labels) == 1:
            fn_labels = np.repeat(np.asarray(labels), num_signals, axis=0)
        elif len(labels) == num_signals:
            fn_labels = np.asarray(labels)
        else:
            # Unsure if this will allow choosing from selection, tiled repeat, or error
            raise NotImplementedError
    if fn_labels.shape != (num_signals, constants.LABEL_LENGTH):
        raise ValueError(
            f'{fn_labels.shape} != {(num_signals, constants.LABEL_LENGTH)}')
    return model.predict(
        (tf.random.normal(shape=(num_signals, LATENT_DIM)), fn_labels))


def save_array(arr, filename: str, overwrite=False):
    """ Save array of generated signals.
        :param: arr -
        :param: filename - str
        :return: None"""
    if not filename.endswith(('.npy', '.txt', '.gz')):
        raise NotImplementedError
    if os.path.isdir(filename):
        raise IsADirectoryError(f'Cannot write array to a directory.')
    if not overwrite and os.path.isfile(filename):
        raise FileExistsError(
            f'Writing to path would overwrite file. Path {filename}')
    if filename.endswith('.npy'):
        np.save(filename, arr)
    elif filename.endswith('.txt') or filename.endswith('.gz'):
        np.savetxt(filename, arr)


def is_valid_label(labels):
    """ Helper function to determine if labels can be used."""
    if labels is None:
        return None
    if not isinstance(labels, str):
        raise TypeError('labels must be a string.')
    # try int, ndarray, then file, else raise error
    parsed_labels = None
    try:
        parsed_labels = int(labels)
        return parsed_labels
    except ValueError:
        # should do log here but will only pass
        pass
    try:
        parsed_labels = np.fromstring(
            labels, count=constants.LABEL_LENGTH, sep=',')
        if not np.all(np.isin(parsed_labels, (0, 1))):
            raise ValueError('All values must be 0 or 1.')
        return parsed_labels
    except ValueError:
        pass
    if os.path.isfile(labels):
        if labels.endswith('.npy'):
            return np.load(labels)
        if labels.endswith('.txt'):
            return np.loadtxt(labels, delimiter=',')
        raise NotImplementedError('Cannot handle files yet.')
    raise ValueError(f'The label could not be parsed. label: {labels}')


def parse():
    """ Command line parser for program."""
    # output file
    # model file
    # num signals
    # overwrite flag
    generator_path = os.path.join(os.curdir, GENERATOR_NAME)  # this may change with changes to directory
    parser = argparse.ArgumentParser(description='Program for production of synthesized data.')
    parser.add_argument('--model', help="The file path to the model's directory", default=generator_path)
    parser.add_argument('--n', '--number_of_signals', help='Number of signals to generate', type=int, default='1')
    parser.add_argument(
        '--labels', help='Labels to use for generating signals. Can be integer from [0, 2], array string, or path to '
                         'file. for integer: 0 for single class, 1 for multiclass, 2 for both. The array string must '
                         'be comma separated i.e. 0, 0, 0, 1. File to load must be .npy or .txt. Defaults to single '
                         'class label if option not included.')
    parser.add_argument(
        '--output',
        help='Name of output file to save signals. Can be .txt or .npy files', default='output.txt')
    parser.add_argument(
        '-overwrite', '-w', action='store_true',
        help='Flag to allow overwriting of output file.')
    args = parser.parse_args()
    print(args)
    signals = generate_data(args.n, args.model, is_valid_label(args.labels))
    if args.output is not None:
        save_array(signals, args.output, args.overwrite)


def main():
    parse()


if __name__ == '__main__':
    main()
