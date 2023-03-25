"""
This module transforms historical data set
of stock price evolutions into structured dataset
of features and target that can be used to build model.
"""
import pandas as pd
import numpy as np


def create_features_and_target_split(df, rol_freq):
    """
    It takes input dataframe and creates rolling features
    dataset with specified frequency and also creates
    target which is the next day values.

    Args:
        df (pd.DataFrame):
        rol_freq (int): number of days consider to predict target

    Returns:
        np.array()
    """
    features = []
    target = []
    for i in range(rol_freq, len(df)):
        features.append(df.iloc[i - rol_freq:i, 0:df.shape[1]])
        target.append(df.iloc[i:i + 1, 0:df.shape[1]])

    return features, target