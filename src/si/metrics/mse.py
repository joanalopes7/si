import numpy as np

def mse (y_true: np.array, y_pred:np.array) -> float:
    """
    ddd
    """
    return (np.sum(y_true-y_pred)**2) / len(y_true)