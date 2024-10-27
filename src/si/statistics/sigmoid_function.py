import numpy as np

def sigmoid_function (x:np.array) -> float :
    """
    Calculate the sigmoid function for the input values.

    Parameters:
    X(array): Input values.

    Returns:
    array: The sigmoid of the input values.
    """
    
    return 1 / (1 + np.exp(-x))
