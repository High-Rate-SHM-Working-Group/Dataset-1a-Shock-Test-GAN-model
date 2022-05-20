import os

import matplotlib.pyplot as plt

import constants
from figure_makers.plot_signal import plot_signal, hello_world

from main import generate_data


def single_signal_plot():
    signal = generate_data(1, constants.GENERATOR_NAME, labels=0)[0]
    fig = plot_signal(signal)
    return fig


def main():
    single_signal_plot()
    plt.savefig(os.path.join(os.curdir, 'figures', 'single_signal.pdf'))
    plt.savefig(os.path.join(os.curdir, 'figures', 'single_signal.png'))
    plt.show()


if __name__ == '__main__':
    main()
