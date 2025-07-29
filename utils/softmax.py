from typing import Union, Sequence
import numpy as np


def softmax(x: Union[Sequence[float], np.ndarray]) -> np.ndarray:
    """
    Compute the softmax of a 1D array or list of floats.

    Args:
        x: A list or 1D numpy array of floats.

    Returns:
        A numpy array containing the softmax probabilities.
    """
    x = np.array(x, dtype=np.float64)
    x_max = np.max(x)  # For numerical stability
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x)
