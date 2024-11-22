import numpy as np

from si.metrics.mse import mse


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the Root Mean Squared Error (RMSE) between the true values and predicted values.

    Parameters
    ----------
    y_true: np.ndarray
        The true labels or actual values in the dataset.
    y_pred: np.ndarray
        The predicted labels or values from the model.

    Returns
    -------
    rmse: float
        The Root Mean Squared Error (RMSE) between the true and predicted values.
        It represents the square root of the average squared differences between the predicted and actual values.
    """
    return np.sqrt(mse(y_true, y_pred))