"""
This module contains methods for data transformations
mostly huber location estimator which is a general
version of combination of median or mean if the underlying
distribution has outliers.
"""

import numpy as np


def huber_location_estimator(x, c=1.345):
    """
    Computes huber location estimator

    Args:
        x: numpy array
        c: tuning constant

    Returns:
        float
    """
    median = np.median(x)
    mad = np.median(np.abs(x - median))
    k = c * mad
    w = np.zeros_like(x)
    w[np.abs(x - median) <= k] = 1
    estimator = np.sum(w * x) / np.sum(w)

    return estimator
