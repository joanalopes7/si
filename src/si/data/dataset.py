from typing import Tuple, Sequence, Union

import numpy as np
import pandas as pd


class Dataset:
    def __init__(self, X: np.ndarray, y: np.ndarray = None, features: Sequence[str] = None, label: str = None) -> None:
        """
        Dataset represents a tabular dataset for single output classification.

        Parameters
        ----------
        X: numpy.ndarray (n_samples, n_features)
            The feature matrix
        y: numpy.ndarray (n_samples, 1)
            The label vector
        features: list of str (n_features)
            The feature names
        label: str (1)
            The label name
        """
        if X is None:
            raise ValueError("X cannot be None")
        if y is not None and len(X) != len(y):
            raise ValueError("X and y must have the same length")
        if features is not None and len(X[0]) != len(features):
            raise ValueError("Number of features must match the number of columns in X")
        if features is None:
            features = [f"feat_{str(i)}" for i in range(X.shape[1])]
        if y is not None and label is None:
            label = "y"
        self.X = X
        self.y = y
        self.features = features
        self.label = label

    def shape(self) -> Tuple[int, int]:
        """
        Returns the shape of the dataset
        Returns
        -------
        tuple (n_samples, n_features)
        """
        return self.X.shape

    def has_label(self) -> bool:
        """
        Returns True if the dataset has a label
        Returns
        -------
        bool
        """
        return self.y is not None

    def get_classes(self) -> np.ndarray:
        """
        Returns the unique classes in the dataset
        Returns
        -------
        numpy.ndarray (n_classes)
        """
        if self.has_label():
            return np.unique(self.y)
        else:
            raise ValueError("Dataset does not have a label")

    def get_mean(self) -> np.ndarray:
        """
        Returns the mean of each feature
        Returns
        -------
        numpy.ndarray (n_features)
        """
        return np.nanmean(self.X, axis=0)

    def get_variance(self) -> np.ndarray:
        """
        Returns the variance of each feature
        Returns
        -------
        numpy.ndarray (n_features)
        """
        return np.nanvar(self.X, axis=0)

    def get_median(self) -> np.ndarray:
        """
        Returns the median of each feature
        Returns
        -------
        numpy.ndarray (n_features)
        """
        return np.nanmedian(self.X, axis=0)

    def get_min(self) -> np.ndarray:
        """
        Returns the minimum of each feature
        Returns
        -------
        numpy.ndarray (n_features)
        """
        return np.nanmin(self.X, axis=0)

    def get_max(self) -> np.ndarray:
        """
        Returns the maximum of each feature
        Returns
        -------
        numpy.ndarray (n_features)
        """
        return np.nanmax(self.X, axis=0)

    def summary(self) -> pd.DataFrame:
        """
        Returns a summary of the dataset
        Returns
        -------
        pandas.DataFrame (n_features, 5)
        """
        data = {
            "mean": self.get_mean(),
            "median": self.get_median(),
            "min": self.get_min(),
            "max": self.get_max(),
            "var": self.get_variance()
        }
        return pd.DataFrame.from_dict(data, orient="index", columns=self.features)

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, label: str = None):
        """
        Creates a Dataset object from a pandas DataFrame

        Parameters
        ----------
        df: pandas.DataFrame
            The DataFrame
        label: str
            The label name

        Returns
        -------
        Dataset
        """
        if label:
            X = df.drop(label, axis=1).to_numpy()
            y = df[label].to_numpy()
        else:
            X = df.to_numpy()
            y = None

        features = df.columns.tolist()
        return cls(X, y, features=features, label=label)

    def to_dataframe(self) -> pd.DataFrame:
        """
        Converts the dataset to a pandas DataFrame

        Returns
        -------
        pandas.DataFrame
        """
        if self.y is None:
            return pd.DataFrame(self.X, columns=self.features)
        else:
            df = pd.DataFrame(self.X, columns=self.features)
            df[self.label] = self.y
            return df

    @classmethod
    def from_random(cls,
                    n_samples: int,
                    n_features: int,
                    n_classes: int = 2,
                    features: Sequence[str] = None,
                    label: str = None):
        """
        Creates a Dataset object from random data

        Parameters
        ----------
        n_samples: int
            The number of samples
        n_features: int
            The number of features
        n_classes: int
            The number of classes
        features: list of str
            The feature names
        label: str
            The label name

        Returns
        -------
        Dataset
        """
        X = np.random.rand(n_samples, n_features)
        y = np.random.randint(0, n_classes, n_samples)
        return cls(X, y, features=features, label=label)
    
    def dropna(self) -> 'Dataset':
        """
        Remove all samples containing at least one null value (NaN) from the dataset.

        Returns
        -------
        Self: The modified Dataset object without NaN values in any independent feature/variable.
        """
        samples_without_na = ~np.isnan(self.X).any(axis=1)
        self.X = self.X[samples_without_na]
        self.y = self.y[samples_without_na]
        return self
    

    def fillna(self, value: Union[float, str]) -> 'Dataset':
        """
        Fills NaN values in the dataset with a specified value.

        Depending on the argument passed to the `value` parameter, NaN values in `self.X` can be replaced by:
        - A specific value (float).
        - The mean of each column (mean).
        - The median of each column (median).

        The method returns the modified Dataset object with NaN values filled according to the provided value.

        Parameters
        ----------
        value : Union[float, str]
            The value used to fill NaN values. It can be:
            - A float, which will replace all NaN values.
            - The string "mean", to replace NaNs with the mean of each column of self.X.
            - The string "median", to replace NaNs with the median of each column of self.X.

        Returns
        -------
        self : Dataset
            The modified Dataset object with NaN values filled according to the specified value.

        Raises
        ------
        ValueError
            If the value parameter is not of type float, "mean", or "median", an error will be raised.
        """

        if isinstance(value, float):
            self.X = np.where(np.isnan(self.X), value, self.X)

        elif value == "mean":
            means = self.get_mean()
            for i in range(self.X.shape[1]):
                self.X[:, i] = np.where(np.isnan(self.X[:, i]), means[i], self.X[:, i])

        elif value == "median":
            medians = self.get_median()
            for i in range(self.X.shape[1]):
                self.X[:, i] = np.where(np.isnan(self.X[:, i]), medians[i], self.X[:, i])
        else:
            raise ValueError("It is not possible to replace with the indicated value.")

        return self

    def remove_by_index(self, index:int) -> 'Dataset':
        """
        Removes the sample at the specified index from the dataset.

        Parameters
        ----------
        index : int
            The index of the sample to be removed from the dataset.

        Returns
        -------
        self : Dataset
            The modified dataset object with the sample removed.
        
        Raises
        ------
        IndexError
            If the specified index is out of bounds (less than 0 or greater than the number of samples in the dataset).
        """
        if index < 0 or index >= len(self.X):
            raise IndexError("Index out of limites.")
        
        self.X = np.delete(self.X, index, axis=0)

        if self.y is not None:
            self.y = np.delete(self.y, index, axis=0)

        return self




if __name__ == '__main__':
    X = np.array([[1, 2, 3], [4, np.nan, 6], [7, 8, 9]])
    y = np.array([1, 2, 3])
    features = np.array(['a', 'b', 'c'])
    label = 'y'
    dataset = Dataset(X, y, features, label)
    print("Before dropna:")
    print("X:\n", dataset.X)
    print("y:\n", dataset.y)

    dataset.dropna()

    print("\nAfter dropna:")
    print("X:\n", dataset.X)
    print("y:\n", dataset.y)
    print("\n", dataset.shape())
    print(dataset.has_label())
    print(dataset.get_classes())
    print(dataset.get_mean())
    print(dataset.get_variance())
    print(dataset.get_median())
    print(dataset.get_min())
    print(dataset.get_max())
    print(dataset.summary())

print("\n")


if __name__ == '__main__':
    X = np.array([[1, 2], [np.nan, 4], [5, np.nan]])
    y = np.array([1, 2, 3])
    features = np.array(['a', 'b'])
    dataset = Dataset(X, y, features)

    print("Before fillna:")
    print("X:\n", dataset.X)
    print("y:\n", dataset.y)

    dataset.fillna("mean")
    print("\nDataset after replace NaN by the average")
    print(dataset.X)

    dataset.fillna("median")
    print("Dataset after replace NaN by the median:")
    print(dataset.X)


    dataset.fillna(0.0)
    print("Dataset after replace NaN by float:")
    print(dataset.X)

    
if __name__ == "__main__":
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([10, 20, 30])
    features = np.array(['a', 'b'])

    dataset = Dataset(X, y, features)

    print("X:\n", dataset.X)
    print("y:\n", dataset.y)

    dataset.remove_by_index(1)

    print("\nDataset after removing sample idx 1:")
    print("X:\n", dataset.X)
    print("y:\n", dataset.y)   