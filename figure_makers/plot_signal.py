""" Signal figure_makers functions.
    Zhymir Thompson 2022"""
import matplotlib.pyplot as plt
import numpy as np


def plot_signal(signal: np.ndarray):
    """ Plot signal in graph."""
    fig = plt.figure()
    plt.plot(signal)
    return fig


def hello_world():
    print('hello world!')

