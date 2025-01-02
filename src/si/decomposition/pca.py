import numpy as np
from si.base.transformer import Transformer
from si.data.dataset import Dataset

class PCA(Transformer):
    """
    Principal Component Analysis (PCA) using eigenvalue decomposition of the covariance matrix.
    
    Parameters
    ----------
    n_components : int
        Number of principal components to keep.
    
    Attributes
    ----------
    mean : ndarray of shape (n_features,)
        Mean of each feature in the training set.
    components : ndarray of shape (n_components, n_features)
        Principal axes in feature space, representing the directions of maximum variance in the data.
    explained_variance : ndarray of shape (n_components,)
        The amount of variance explained by each of the selected components.
    """

    def __init__(self, n_components):
        """
        Initialize PCA with the number of components.
        
        Parameters
        ----------
        n_components : int
            Number of principal components to keep.
        """
        self.n_components = n_components
        self.mean = None
        self.components = None
        self.explained_variance = None

    def _fit(self, dataset: Dataset):
        """
        Fit the model with X by estimating the mean, principal components, and explained variance.
        
        Parameters
        ----------
        dataset : Dataset
           A labeled dataset
        """ 
        #Center the data
        X=dataset.X
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean

        #Calculate the covariance matrix of the centered data
        covariance_matrix = np.cov(X_centered, rowvar=False)

        #Perform eigenvalue decomposition on the covariance matrix
        eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
        sorted_indices = np.argsort(eigenvalues)[::-1]
        sorted_eigenvalues = eigenvalues[sorted_indices]
        sorted_eigenvectors = eigenvectors[:, sorted_indices]

        #Infer the Principal Components
        self.components = sorted_eigenvectors[:, :self.n_components]

        #Infer the Explained Variance
        total_variance = np.sum(sorted_eigenvalues)
        self.explained_variance = sorted_eigenvalues[:self.n_components] / total_variance

    def _transform(self, dataset):
        """
        Apply dimensionality reduction to dataset using the fitted principal components.
        
        Parameters
        ----------
        dataset : Dataset
           A labeled dataset
        
        Returns
        -------
        dataset_reduced : ndarray of shape (n_samples, n_components)
            Transformed data.
        """
        X = dataset.X
        X_centered = X - self.mean
        dataset_reduced = np.dot(X_centered, self.components)
        return dataset_reduced