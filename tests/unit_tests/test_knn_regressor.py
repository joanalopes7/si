from unittest import TestCase

import numpy as np


from datasets import DATASETS_PATH

import os
from si.io.csv_file import read_csv
from si.models.knn_regressor import KNNRegressor
from si.model_selection.split import train_test_split
from si.metrics.rmse import rmse
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from math import sqrt


class TestKNNRegressor(TestCase):

    def setUp(self):
        self.csv_file = os.path.join('datasets', 'cpu', 'cpu.csv')  
        self.dataset = read_csv(filename=self.csv_file, features=True, label=True)


    def test_fit(self):
        knn = KNNRegressor(k=3)
        knn.fit(self.dataset)
        self.assertTrue(np.all(self.dataset.X == knn.dataset.X))
        self.assertTrue(np.all(self.dataset.y == knn.dataset.y))

    def test_predict(self):
        knn = KNNRegressor(k=3)
        train_dataset, test_dataset = train_test_split(self.dataset)
        knn.fit(train_dataset)
        predictions = knn.predict(test_dataset)
        self.assertEqual(predictions.shape[0], test_dataset.y.shape[0])

    def test_score(self):
        knn = KNNRegressor(k=3)
        train_dataset, test_dataset = train_test_split(self.dataset, test_size=0.2)

        knn.fit(train_dataset)
        predictions_custom = knn.predict(test_dataset)
        my_rsme = sqrt(mean_squared_error(test_dataset.y, predictions_custom))


        sklearn_knn = KNeighborsRegressor(n_neighbors=3)
        sklearn_knn.fit(train_dataset.X, train_dataset.y)
        predictions_sklearn = sklearn_knn.predict(test_dataset.X)
        rmse_sklearn = sqrt(mean_squared_error(test_dataset.y, predictions_sklearn))

        self.assertAlmostEqual(my_rsme, rmse_sklearn, delta=0.1)

