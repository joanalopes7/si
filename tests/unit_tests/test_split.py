from unittest import TestCase

from datasets import DATASETS_PATH

import os
from si.io.csv_file import read_csv
import numpy as np

from si.model_selection.split import train_test_split
from si.model_selection.split import stratified_train_test_split

class TestSplits(TestCase):

    def setUp(self):
        self.csv_file = os.path.join(DATASETS_PATH, 'iris', 'iris.csv')

        self.dataset = read_csv(filename=self.csv_file, features=True, label=True)

    def test_train_test_split(self):

        train, test = train_test_split(self.dataset, test_size = 0.2, random_state=123)
        test_samples_size = int(self.dataset.shape()[0] * 0.2)
        self.assertEqual(test.shape()[0], test_samples_size)
        self.assertEqual(train.shape()[0], self.dataset.shape()[0] - test_samples_size)

    def test_stratified_train_test_split(self):

        train, test = train_test_split(self.dataset, test_size = 0.2, random_state=123)
        test_samples_size = int(self.dataset.shape()[0] * 0.2)
        self.assertEqual(test.shape()[0], test_samples_size)
        self.assertEqual(train.shape()[0], self.dataset.shape()[0] - test_samples_size)

        _, train_counts = np.unique(train.y, return_counts=True)
        _, test_counts = np.unique(test.y, return_counts=True)
        _, original_counts = np.unique(self.dataset.y, return_counts=True)
        original_ratios = original_counts / self.dataset.shape()[0]
        train_ratios = train_counts / train.shape()[0]
        test_ratios = test_counts / test.shape()[0]

        np.testing.assert_almost_equal(train_ratios, original_ratios, decimal=1)
        np.testing.assert_almost_equal(test_ratios, original_ratios, decimal=1)
        
