from unittest import TestCase

import numpy as np
from sklearn.metrics import mean_squared_error
from si.metrics.accuracy import accuracy
from si.metrics.mse import mse
from si.statistics.sigmoid_function import sigmoid_function
from si.metrics.rmse import rmse

class TestMetrics(TestCase):

    def test_accuracy(self):

        y_true = np.array([0,1,1,1,1,1,0])
        y_pred = np.array([0,1,1,1,1,1,0])

        self.assertTrue(accuracy(y_true, y_pred)==1)

    def test_mse(self):

        y_true = np.array([0.1,1.1,1,1,1,1,0])
        y_pred = np.array([0,1,1.1,1,1,1,0])

        self.assertTrue(round(mse(y_true, y_pred), 3)==0.004)
        
    def test_sigmoid_function(self):

        x = np.array([1.9, 10.4, 75])

        x_sigmoid = sigmoid_function(x)

        self.assertTrue(all(x_sigmoid >= 0))
        self.assertTrue(all(x_sigmoid <= 1))

    def test_rmse(self):
        # Valores reais e previstos
        y_true = np.array([0.1, 1.1, 1, 1, 1, 1, 0])
        y_pred = np.array([0, 1, 1.1, 1, 1, 1, 0])
        
        my_rmse = rmse(y_true, y_pred)
        sklearn_mse = mean_squared_error(y_true, y_pred)
        sklearn_rmse = np.sqrt(sklearn_mse)
        assert np.isclose(my_rmse, sklearn_rmse)
