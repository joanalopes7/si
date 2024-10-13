import numpy as np

def cosine_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Calculate the Cosine distance between a single sample x and multiple samples y.

    Parameters
    ----------
    x : np.ndarray
        A single sample.
    y : np.ndarray
        Multiple samples.

    Returns
    -------
    np.ndarray
        An array containing the Cosine distances between x and the various samples in y.
    """

    dot_product = np.dot(y, x)
    norm_x = np.linalg.norm(x)
    norm_y = np.linalg.norm(y, axis=1)
    cosine_similarity = dot_product / (norm_x * norm_y)
    distance = 1 - cosine_similarity
    
    return distance
