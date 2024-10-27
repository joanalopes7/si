from unittest import TestCase
from si.statistics.sigmoid_function import sigmoid_function
import os
import numpy as np


class TestSigmoidFunction(TestCase):

    def test_sigmoid(self):
        y = np.array([1,2,3]) 

        self.assertTrue(all(sigmoid_function(y)>0))
        self.assertTrue(all(sigmoid_function(y)<1))
