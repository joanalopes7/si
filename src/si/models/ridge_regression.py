from typing import Callable
import numpy as np
from si.base.model import Model
from si.data.dataset import Dataset
from si.metrics.mse import mse

class RidgeRegression(Model):
    
    def __init__(self, l2_penalty: float = 1 , aplha: float = 0.001, 
                 max_iter: int = 100, patience : int = 5 , scale: bool = True , **kwargs):
        """
        Initialize the RidgeRegression.

        Parameters
        ----------
        l2_penalty: int
            L2 regulation parameter.
        alpha : float
            Learning rate.
        Max_iter : 
            Maximium number of iterations.
        Patience :
            Maximum number of iterations without improvement allowed.
        Scale :
            Wheter to scale the data or not.
        """
        # parameters
        super().__init__(**kwargs)
        self.l2_penalty = l2_penalty
        self.aplha = aplha
        self.max_iter = max_iter
        self.patience = patience
        self.scale = scale
    
        # attributes
        self.theta = None
        self.theta_zero = 0
        self.mean = None
        self.std = None
        self.cost_history = {}
    
    def _fit(self, dataset: Dataset ) -> 'RidgeRegression':

        if self.scale:
            self.mean = dataset.get_mean()
            self.std = np.nanstd(dataset.X, axis=0)
            x= (dataset.X - self.mean) / self.std

        else: 
            x = dataset.X

        m, n = dataset.shape()
        self.theta = np.zeros(n)

        i=0
        early_stopping = 0
        while i < self.max_iter and early_stopping < self.patience:
            y_pred = np.dot(self.theta,dataset.X) + self.theta_zero
            gradient = (self.aplha/m) * np.dot((y_pred-dataset.y),x)
            penalty_term_gradient = self.theta * (1-(self.aplha * self.l2_penalty/m))

            self.theta = penalty_term_gradient - gradient
            self.theta_zero = self.theta_zero - gradient
            
            y_pred = np.dot(self.theta,dataset.X) + self.theta_zero
            cost = (np.sum ( ( y_pred - dataset.y) ** 2) + (self.l2_penalty * np.sum(self.theta **2))) / (2*m)

            self.cost_history[i]=cost

            if i>0 and self.cost_history[i-1] < cost:
                early_stopping +=1
            else:
                early_stopping=0
            i += 1

        return self


    def _predict(self, dataset: Dataset):
        pass

    def _score(self, dataset : Dataset, predictions):
        pass

    def cost (self, distance: Callable = mse, y_true: np.array, y_pred:np.array):
        pass