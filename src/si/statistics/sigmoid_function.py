import numpy as np

<<<<<<< HEAD
def sigmoid_function (x:np.array) -> float :
    """
    Calculate the sigmoid function for the input values.

    Parameters:
    X(array): Input values.

    Returns:
    array: The sigmoid of the input values.
    """
    
    return 1 / (1 + np.exp(-x))
=======

def sigmoid_function(X: np.ndarray) -> np.ndarray:
    """
    It returns the sigmoid function of the given input

    Parameters
    ----------
    X: np.ndarray
        The input of the sigmoid function

    Returns
    -------
    sigmoid: np.ndarray
        The sigmoid function of the given input
    """
    return 1 / (1 + np.exp(-X))
>>>>>>> 798823c15b5d67e400b65f8e9be43ee1e13774e5
