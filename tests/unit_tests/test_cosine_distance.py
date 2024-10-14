from unittest import TestCase

import numpy as np
from datasets import DATASETS_PATH

import os
from si.io.csv_file import read_csv

from si.statistics.cosine_distance import cosine_distance

class TestCosineDistance(TestCase):

    def test_cosine_distance(self):
        x = np.array([1, 2, 3])
        y = np.array([[1, 2, 3], [4, 5, 6]])
        our_distance = cosine_distance(x, y)

        from sklearn.metrics.pairwise import cosine_distances
        sklearn_distance = cosine_distances(x.reshape(1, -1), y)
        assert np.allclose(our_distance, sklearn_distance)


