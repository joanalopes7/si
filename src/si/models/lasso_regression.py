#Evaluation: exercise 8 

import numpy as np

from si.base.model import Model
from si.data.dataset import Dataset
from si.metrics.mse import mse


class LassoRegression(Model):
    """
    The LassoRegression is a linear model using the L1 regularization.
    This model solves the linear regression problem using coordinate descent for optimization.
    Parameters
    ----------
    l1_penalty: float
        The L1 regularization parameter
    scale: bool
        Whether to scale the dataset or not

    Attributes
    ----------
    theta: np.array
        The model parameters, namely the coefficients of the linear model.
        For example, x0 * theta[0] + x1 * theta[1] + ...
    theta_zero: float
        The model parameter, namely the intercept of the linear model.
        For example, theta_zero * 1
    mean (np.ndarray): Mean of the dataset for every feature.

    std (np.ndarray): Standard deviation of the dataset for every feature.
    """
    def __init__(self, l1_penalty: float = 1,max_iter: int = 1000, patience: int = 5, scale: bool = True, **kwargs):
        """
        Parameters
        ----------
        l1_penalty: float
            The L1 regularization parameter
        max_iter: int
            The maximum number of iterations
        patience: int
            The number of iterations without improvement before stopping the training
        scale: bool
            Whether to scale the dataset or not
        """
        
        # parameters
        super().__init__(**kwargs) 
        self.l1_penalty=l1_penalty
        self.scale=scale
        self.max_iter = max_iter
        self.patience = patience
        
        # attributes
        self.theta = None
        self.theta_zero = None
        self.mean = None
        self.std = None
        self.cost_history = {}


    def soft_thresholding(self, rho_j, l1_penalty):
        """
        Soft Thresholding operator for Lasso Regression. 
        Used to shrink coefficients towards zero.

        Parameters
        ----------
        rho_j: float
            The residual for feature j.
        l1_penalty: float
            The L1 penalty (regularization strength).

        Returns
        -------
        float
            The updated coefficient for feature j.
        """
        if rho_j < -l1_penalty:
            return (rho_j + l1_penalty)
        elif rho_j > l1_penalty:
            return (rho_j - l1_penalty)
        else:
            return 0


    def _fit(self, dataset: Dataset) -> 'LassoRegression':
        """
        Fit the model to the dataset

        Parameters
        ----------
        dataset: Dataset
            The dataset to fit the model to

        Returns
        -------
        self: LassoRegression
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
        
        #coordinate descent
        while i < self.max_iter and early_stopping < self.patience:
            y_pred = np.dot(X, self.theta) + self.theta_zero

            #iterate over each feature
            for j in range(n):
                # Residuals for feature j, excluding the effect of feature j
                residuals = dataset.y - (X.dot(self.theta)) + self.theta[j] * X[:, j]

                # Calculate r_j (the inner product of X[:, j] and the residuals)
                r_j = np.dot(X[:, j], residuals)

                self.theta[j] = self.soft_thresholding(r_j, self.l1_penalty) / np.dot(X[:, j], X[:, j])

            #update theta_zero
            self.theta_zero = np.mean(dataset.y - np.dot(X, self.theta))

            #compute the cost
            self.cost_history[i] = self.cost(dataset)
            if i > 0 and self.cost_history[i] > self.cost_history[i - 1]:
                early_stopping += 1
            else:
                early_stopping = 0

            i += 1

        return self

    def _predict(self, dataset: Dataset) -> np.ndarray:
        """
        Predict the output of the dataset.

        Parameters
        ----------
        dataset: Dataset
            The dataset to predict the output of.

        Returns
        -------
        np.ndarray
            The predictions of the dataset.
        """
        # Scale the data using the mean and std from the fit method
        if self.scale:
            # Normalizar X usando a mesma média e desvio padrão do treinamento
            X = (dataset.X - self.mean) / self.std
        else:
            X = dataset.X
        return np.dot(X, self.theta) + self.theta_zero

    def _score(self, dataset: Dataset, predictions: np.ndarray) -> float:
        """
        Compute the Mean Square Error of the model on the dataset.

        Parameters
        ----------
        dataset: Dataset
            The dataset to compute the MSE on.

        predictions: np.ndarray
            Predictions

        Returns
        -------
        mse: float
            The Mean Square Error of the model.
        """
        return mse(dataset.y, predictions)
    
    def cost(self, dataset: Dataset) -> float:
        """
        Compute the cost function (J function) of the model on the dataset using L1 regularization

        Parameters
        ----------
        dataset: Dataset
            The dataset to compute the cost function on

        Returns
        -------
        cost: float
            The cost function of the model
        """
        y_pred = self.predict(dataset)
        return (np.sum((y_pred - dataset.y) ** 2) + self.l1_penalty * np.sum(np.abs(self.theta))) / (2 * len(dataset.y))
    


if __name__ == '__main__':
    # import dataset
    from si.data.dataset import Dataset

    # make a linear dataset
    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y = np.dot(X, np.array([1, 2])) + 3
    dataset_ = Dataset(X=X, y=y)

    # fit the model
    model = LassoRegression()
    model.fit(dataset_)

    # get coefs
    print(f"Parameters: {model.theta}")

    # compute the score
    score = model.score(dataset_)
    print(f"Score: {score}")

    # compute the cost
    cost = model.cost(dataset_)
    print(f"Cost: {cost}")

    # predict
    y_pred_ = model.predict(Dataset(X=np.array([[3, 5]])))
    print(f"Predictions: {y_pred_}")