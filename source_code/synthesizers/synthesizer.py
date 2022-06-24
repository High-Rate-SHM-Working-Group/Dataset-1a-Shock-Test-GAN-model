""" Zhymir Thompson 2022"""
import pandas as pd
import os
import numpy as np
from source_code.constants import GENERATOR_NAME, SINGLE_LABELS, MULTI_LABELS, TIME
from source_code.main import generate_data


# This code will break if generator moves
def generate_synthesized_data(num_signals=100, labels=None):
    data = generate_data(num_signals, os.path.join(os.pardir, GENERATOR_NAME), labels)
    return data


def write_data(uni_path=None, multi_path=None):
    # Unimodal
    tests: int = 100
    single_names = ['Accelerometer_1', 'Accelerometer_2', 'Accelerometer_3', 'Accelerometer_4']
    time = np.asarray([TIME])
    num_singles = len(SINGLE_LABELS)
    labels = np.asarray(SINGLE_LABELS)
    path = uni_path if uni_path is not None else os.curdir
    # if not os.path.isdir(path):
    #     os.makedirs(path)
    for test_idx in range(1, tests + 1):
        data = generate_synthesized_data(num_singles, labels)
        frame = pd.DataFrame(data=np.concatenate((time, data), axis=0).transpose((1, 0)), columns=['time']+single_names)
        frame.to_csv(os.path.join(path, f'test_{test_idx}.txt'), sep='\t', index=False)
    # Multimodal
    path = multi_path if multi_path is not None else os.curdir
    multi_names = [
        'Accelerometer_1,_2', 'Accelerometer_1,_3', 'Accelerometer_1,_4',
        'Accelerometer_3,_4', 'Accelerometer_2,_3', 'Accelerometer_2,_4',
        'Accelerometer_1,_2,_3', 'Accelerometer_1,_2,_4', 'Accelerometer_1,_3,_4',
        'Accelerometer_2,_3,_4', 'Accelerometer_1,_2,_3,_4']
    num_multi = len(MULTI_LABELS)
    labels = np.asarray(MULTI_LABELS)
    # if not os.path.isdir(path):
    #     os.makedirs(path)
    for test_idx in range(1, tests + 1):
        data = generate_synthesized_data(num_multi, labels)
        frame = pd.DataFrame(data=np.concatenate((time, data), axis=0).transpose((1, 0)), columns=['time']+multi_names)
        frame.to_csv(os.path.join(path, f'test_{test_idx}.txt'), sep='\t', index=False)


if __name__ == '__main__':
    folder = os.path.join(os.pardir, os.pardir, 'synthesized_data')
    uni_path = os.path.join(folder, 'uni-modal')
    multi_path = os.path.join(folder, 'multi-modal')
    write_data(uni_path=uni_path, multi_path=multi_path)
