"""
This module transforms historical data set
of stock price evolutions into structured dataset
of features and target that can be used to build model.
"""
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler


def create_features_and_target_split(df, rol_freq):
    """
    It takes input dataframe and creates rolling features
    dataset with specified frequency and also creates
    target which is the next day values.

    Args:
        df (pd.DataFrame):
        rol_freq (int): number of days consider to predict target

    Returns:
        pd.DataFrame
    """
    ls = list(range(1, rol_freq + 1)) + ['target']
    dt_preprocess = pd.DataFrame(columns=ls, index=df.index)
    for i in range(rol_freq, len(df)):
        features_ls = df.iloc[i - rol_freq:i, 0:df.shape[1]]['daily_returns'].to_list()
        target_ls = df.iloc[i:i + 1, 0:df.shape[1]]['daily_returns'].to_list()
        dt_preprocess.iloc[i] = features_ls + target_ls

    dt_preprocess = dt_preprocess.dropna()

    return dt_preprocess


def standardize_and_limit_outliers_returns(df):
    """
    Compute daily returns, standardize the returns and
    limit the returns if there are outliers in both
    positive and negative side.

    Args:
        df (pd.DataFrame) : historical dataset

    Returns:
        pd.DataFrame
    """

    return df


def train_validation_test_split(stock_name, df_hist, **kwargs):
    """
    Split data into training and validation test. Note that as
    the data is time series, so, split will not be random. Keep
    1 year of data for validation purpose and last 1 year of data
    for testing purpose and rest all for training.

    Args:
        stock_name (str): stock name
        df_hist (pd.DataFrame): historical complete data
        **kwargs: training and validation test split

    Returns:
        training and validation dataset
    """
    df_hist['daily_returns'] = df_hist['Adj Close'].pct_change()
    df_hist = df_hist.dropna()
    df_features, df_target = create_features_and_target_split(pd.DataFrame(df_hist['daily_returns']), 60)

    # split the original dataset into 3 sets
    df_training = df_hist[df_hist.index <= kwargs['training_end']]
    df_validation = df_hist[
        (df_hist.index > kwargs['training_end']) & (df_hist.index <= kwargs['validation_end'])]
    df_test = df_hist[df_hist.index > kwargs['validation_end']]

    # create rolling dataset

    return
