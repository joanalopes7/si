from unittest import TestCase

import numpy as np

from datasets import DATASETS_PATH
from si.io.csv_file import read_csv
import os
from si.decomposition.pca import PCA

class TestPCA(TestCase):

    def setUp(self):
        self.csv_file = os.path.join(DATASETS_PATH, 'iris', 'iris.csv')
        self.dataset = read_csv(filename=self.csv_file, features=True, label=True)
        self.n_components = 2
        self.pca = PCA(self.n_components)

    def test_fit(self):
        self.pca.fit(self.dataset)     
        self.assertTrue(np.allclose(self.pca.mean, np.mean(self.dataset.X, axis=0)))
        self.assertEqual(self.pca.components.shape, (self.dataset.X.shape[1], self.n_components))
        self.assertEqual(len(self.pca.explained_variance), self.n_components)
        self.assertAlmostEqual(np.sum(self.pca.explained_variance), 1.0, delta=0.05)

    def test_transform(self):
        self.pca.fit(self.dataset)
        X_reduced = self.pca.transform(self.dataset)
        self.assertEqual(X_reduced.shape[0], self.dataset.X.shape[0])
        original_variance = np.var(self.dataset.X, axis=0).sum()
        reduced_variance = np.var(X_reduced, axis=0).sum()
        self.assertAlmostEqual(original_variance, reduced_variance, delta=0.2)





