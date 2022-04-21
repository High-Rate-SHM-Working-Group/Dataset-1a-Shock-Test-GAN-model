import os
import unittest
from unittest import TestCase
# TODO move imports into Test suite
import constants
import numpy as np
from main import random_labels, generate_data, is_valid_label


class Test(TestCase):

    def setUp(self) -> None:
        self.num_signals = 200
        self.gen_shape = (self.num_signals, constants.LABEL_LENGTH)
        self.model_output_shape = (
            self.num_signals, constants.MODEL_OUTPUT_LENGTH)  # Loaded model output shape
        self.generator_path = os.path.join(os.curdir, constants.GENERATOR_NAME)

    # def test_random_labels(self):
    #     pass

    def test_random_labels_non_positive_num_signals(self):
        with self.assertRaises(ValueError):
            random_labels(-1, 0)
        with self.assertRaises(ValueError):
            random_labels(0, 0)

    def test_random_labels_incorrect_label_type(self):
        with self.assertRaises(ValueError):
            random_labels(0, -1)
        with self.assertRaises(ValueError):
            random_labels(0, 3)

    def test_random_labels_label_zero(self):
        labels = random_labels(self.num_signals, 0)
        self.assertTrue(
            np.all(np.isin(labels, constants.SINGLE_LABELS)))
        self.assertEqual(self.gen_shape, labels.shape)

    def test_random_labels_label_one(self):
        labels = random_labels(self.num_signals, 1)
        self.assertTrue(
            np.all(np.isin(labels, constants.MULTI_LABELS)))
        self.assertEqual(self.gen_shape, labels.shape)

    def test_random_labels_label_two(self):
        labels = random_labels(self.num_signals, 2)
        self.assertTrue(
            np.all(np.isin(labels, constants.LABEL_CHOICES)))
        self.assertEqual(self.gen_shape, labels.shape)

    # def test_generate_data(self):
    #     pass

    def test_generate_data_type_error(self):
        with self.assertRaises(TypeError):
            generate_data(self.num_signals, self.generator_path, '0')

    def test_generate_data_value_error(self):
        with self.assertRaises(ValueError):
            generate_data(self.num_signals, self.generator_path, 3)

    def test_generate_data_not_implemented_error(self):
        with self.assertRaises(NotImplementedError):
            generate_data(
                self.num_signals, self.generator_path,
                [[1, 1, 1, 1]]*(self.num_signals - 2))

    def test_generate_data_int_label(self):
        int_label = 0
        self.assertEqual(
            self.model_output_shape,
            generate_data(
                self.num_signals, self.generator_path, int_label).shape)

    def test_generate_data_single_list_label(self):
        single_list_label = [[1, 1, 1, 1]]
        self.assertEqual(
            self.model_output_shape,
            generate_data(
                self.num_signals,
                self.generator_path, single_list_label).shape)

    def test_generate_data_full_list_label(self):
        full_list_label = [[1, 1, 1, 1]] * self.num_signals
        self.assertEqual(
            self.model_output_shape,
            generate_data(
                self.num_signals,
                self.generator_path, full_list_label).shape)

    def test_save_array(self):
        self.skipTest('Not written yet.')

    def test_is_valid_label(self):
        self.skipTest('Not written yet.')

    def test_is_valid_label_none(self):
        self.assertIsNone(is_valid_label(None))

    def test_is_valid_label_str(self):
        with self.assertRaises(ValueError):
            is_valid_label('test_string')

    def test_is_valid_label_str_int(self):
        # for idx in range(30):
        #     with self.subTest(idx):
        #         self.assertEqual(str(idx), idx)
        self.assertEqual(is_valid_label('1'), 1)

    def test_is_valid_label_str_np_array(self):
        self.assertEqual(np.array([0, 0, 1, 0]), is_valid_label('0, 0, 1, 0'))

    def test_is_valid_label_str_np_array_no_commas(self):
        with self.assertRaises(ValueError):
            is_valid_label('0 0 1 0')

    def test_is_valid_label_str_filename(self):
        self.skipTest('Not written yet.')

    def test_parse(self):
        self.skipTest('Not written yet.')


if __name__ == '__main__':
    unittest.main()
