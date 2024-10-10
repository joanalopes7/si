import unittest

import numpy as np

from si.data.dataset import Dataset


class TestDataset(unittest.TestCase):

    def test_dataset_construction(self):

        X = np.array([[1, 2, 3], [4, 5, 6]])
        y = np.array([1, 2])

        features = np.array(['a', 'b', 'c'])
        label = 'y'
        dataset = Dataset(X, y, features, label)

        self.assertEqual(2.5, dataset.get_mean()[0])
        self.assertEqual((2, 3), dataset.shape())
        self.assertTrue(dataset.has_label())
        self.assertEqual(1, dataset.get_classes()[0])
        self.assertEqual(2.25, dataset.get_variance()[0])
        self.assertEqual(1, dataset.get_min()[0])
        self.assertEqual(4, dataset.get_max()[0])
        self.assertEqual(2.5, dataset.summary().iloc[0, 0])

    def test_dataset_from_random(self):
        dataset = Dataset.from_random(10, 5, 3, features=['a', 'b', 'c', 'd', 'e'], label='y')
        self.assertEqual((10, 5), dataset.shape())
        self.assertTrue(dataset.has_label())

    def test_dropna(self):
        X = np.array([[1, 2], [np.nan, 4], [5, 6]])
        y = np.array([1, 2, 3])

        features = np.array(['a', 'b'])
        label = 'y'
        dataset = Dataset(X, y, features, label)

        modif_dataset=dataset.dropna()
        assert np.array_equal(modif_dataset.X, np.array([[1, 2], [5, 6]])), "Test failed: row with NaN was not removed."
        assert np.array_equal(modif_dataset.y, np.array([1, 3])), "Test failed: corresponding label was not removed."

    def test_fillna_float(self):
        X = np.array([[1, 2], [np.nan, 4], [5, np.nan]])
        y = np.array([1, 2, 3])
        
        features = np.array(['a', 'b'])
        dataset = Dataset(X, y, features)

        modif_dataset_float=dataset.fillna(0.0)
        expected_X = np.array([[1, 2], [0, 4], [5, 0]])
        np.testing.assert_array_equal(modif_dataset_float.X, expected_X)

    def test_fillna_mean(self):
        X = np.array([[1, 2], [np.nan, 4], [5, np.nan]])
        y = np.array([1, 2, 3])
        
        features = np.array(['a', 'b'])
        dataset = Dataset(X, y, features)

        means = dataset.get_mean()
        modif_dataset_mean=dataset.fillna("mean")
        expected_X = np.array([
            [1, 2],
            [means[0], 4],
            [5, means[1]]])
        np.testing.assert_array_almost_equal(modif_dataset_mean.X, expected_X)

    def test_fillna_median(self):
        X = np.array([[1, 2], [np.nan, 4], [5, np.nan]])
        y = np.array([1, 2, 3])
        
        features = np.array(['a', 'b'])
        dataset = Dataset(X, y, features)

        modif_dataset_median=dataset.fillna("median")
        median_values = dataset.get_median()
        expected_X = np.array([
            [1, 2],
            [median_values[0], 4],
            [5, median_values[1]]])
        np.testing.assert_array_almost_equal(modif_dataset_median.X, expected_X)

    def test_remove_by_index(self):
        X = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([1, 2, 3])
        features = np.array(['a', 'b'])
        dataset = Dataset(X, y, features)

        modif_dataset=dataset.remove_by_index(1)  
        expected_X = np.array([[1, 2], [5, 6]])
        expected_y = np.array([1, 3])
        
        np.testing.assert_array_equal(modif_dataset.X, expected_X)
        np.testing.assert_array_equal(modif_dataset.y, expected_y)

    def test_remove_by_index_invalid(self):
        X = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([1, 2, 3])
        features = np.array(['a', 'b'])
        dataset = Dataset(X, y, features)
    
        with self.assertRaises(IndexError):
            dataset.remove_by_index(5)  