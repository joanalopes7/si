import numpy as np

from si.base.model import Model
from si.data.dataset import Dataset
from si.statistics.sigmoid_function import sigmoid_function


class LogisticRegression(Model):
        """

        Parameters
        ----------
        l2_penalty: float
            The L2 regularization parameter
        alpha: float
            The learning rate
        max_iter: int
            The maximum number of iterations
        patience: int
            The number of iterations without improvement before stopping the training
        scale: bool
            Whether to scale the dataset or not
        """

        def __init__(self, l2_penalty: float = 1, alpha: float = 0.001, max_iter: int = 1000, patience: int = 5,
                 scale: bool = True, **kwargs):
        
            # parameters
            super().__init__(**kwargs)
            self.l2_penalty = l2_penalty
            self.alpha = alpha
            self.max_iter = max_iter
            self.patience = patience
            self.scale = scale

            # attributes
            self.theta = None
            self.theta_zero = None
            self.mean = None
            self.std = None
            self.cost_history = {}

        def _fit(self, dataset: Dataset) -> 'LogisticRegression':
            """
            Fit the model to the dataset

            Parameters
            ----------
            dataset: Dataset
                The dataset to fit the model to

            Returns
            -------
            self: LogisticRegression
                The fitted model
            """
            if self.scale:
                # compute mean and std
                self.mean = np.nanmean(dataset.X, axis=0)
                self.std = np.nanstd(dataset.X, axis=0)
                # scale the dataset
                X = (dataset.X - self.mean) / self.std
            else:
                X = dataset.X

            m, n = dataset.shape()

            # initialize the model parameters
            self.theta = np.zeros(n)
            self.theta_zero = 0

            i = 0
            early_stopping = 0
            # gradient descent
            while i < self.max_iter and early_stopping < self.patience:
                # predicted y
                y_pred = sigmoid_function(np.dot(X, self.theta) + self.theta_zero)

                # computing and updating the gradient with the learning rate
                gradient = (self.alpha / m) * np.dot(y_pred - dataset.y, X)

                # computing the penalty
                penalization_term = self.theta * (1 - self.alpha * (self.l2_penalty / m))

                # updating the model parameters
                self.theta = penalization_term - gradient
                self.theta_zero = self.theta_zero - (self.alpha * (1 / m)) * np.sum(y_pred - dataset.y)

                # compute the cost
                self.cost_history[i] = self.cost(dataset)
                if i > 0 and self.cost_history[i] > self.cost_history[i - 1]:
                    early_stopping += 1
                else:
                    early_stopping = 0
                i += 1

            return self
        
        def _predict(self, dataset: Dataset) -> np.array:
            """
            Predict the output of the dataset

            Parameters
            ----------
            dataset: Dataset
                The dataset to predict the output of

            Returns
            -------
            predictions: np.array
                The predictions of the dataset
            """
            X = (dataset.X - self.mean) / self.std if self.scale else dataset.X
            prob_values=sigmoid_function(np.dot(X, self.theta) + self.theta_zero)

