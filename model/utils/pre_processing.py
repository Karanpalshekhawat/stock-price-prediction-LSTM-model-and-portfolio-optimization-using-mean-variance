"""
This module contains methods used for train test split,
creating structured dataset of features and target variable that
can be common and can be reused across all model.
"""

import pandas as pd
import numpy as np
import tulipy as ti

from sklearn.preprocessing import StandardScaler


def create_features_and_target_split(df, col, rol_freq):
    """
    It takes input dataframe and creates rolling features
    dataset with specified frequency and also creates
    target which is the next day values.

    Args:
        df (pd.DataFrame): historical dataset of either daily returns or closing prices
        col (str): features that will be used for prediction, either daily returns or closing prices
        rol_freq (int): number of days consider to predict target

    Returns:
        pd.DataFrame
    """
    ls = list(range(1, rol_freq + 1)) + ['target']
    dt_preprocess = pd.DataFrame(columns=ls, index=df.index)
    for i in range(rol_freq, len(df)):
        features_ls = df.iloc[i - rol_freq:i, 0:df.shape[1]][col].to_list()
        target_ls = df.iloc[i:i + 1, 0:df.shape[1]][col].to_list()
        dt_preprocess.iloc[i] = features_ls + target_ls

    dt_preprocess = dt_preprocess.dropna()

    return dt_preprocess


def standardize_and_limit_outliers_returns(dt_model, rol_freq, technical_indicator_features, **kwargs):
    """
    Compute daily returns, standardize the returns and
    limit the returns if there are outliers in both
    positive and negative side.

    Args:
        dt_model (pd.DataFrame) : historical dataset
        rol_freq (int) : past return used
        technical_indicator_features (list) : feature columns for technical indicators
        **kwargs: training and validation test split

    Returns:
        dataframes
    """
    # split the original dataset into 3 sets
    df_training = dt_model[dt_model.index <= kwargs['training_end']]
    df_validation = dt_model[
        (dt_model.index > kwargs['training_end']) & (dt_model.index <= kwargs['validation_end'])]
    X_train = np.concatenate((df_training.iloc[:, :rol_freq].values, df_training.loc[:, technical_indicator_features]),
                             axis=1)
    X_val = np.concatenate(
        (df_validation.iloc[:, :rol_freq].values, df_validation.loc[:, technical_indicator_features]), axis=1)
    Y_train = df_training['target'].values
    Y_val = df_validation['target'].values
    # Normalize the feature vectors in the training set
    scaler = StandardScaler()
    X_train_norm = scaler.fit_transform(X_train)
    X_val_norm = scaler.transform(X_val)

    return X_train_norm, Y_train, X_val_norm, Y_val, scaler


def add_technical_indicators(data):
    """
    This method adds momentum indicators in the
    historical dataset of the stock prices

    Args:
        data (pd.DataFrame): stock price data from yahoo finance

    """
    sma_20 = ti.sma(data['Adj Close'].values, 20)
    sma_50 = ti.sma(data['Adj Close'].values, 50)
    data['MA20'] = [np.nan] * (len(data) - len(sma_20)) + list(sma_20)
    data['MA50'] = [np.nan] * (len(data) - len(sma_50)) + list(sma_50)
    rsi = ti.rsi(data['Adj Close'].values, period=14)
    data['RSI'] = [np.nan] * (len(data) - len(rsi)) + list(rsi)
    ma_cd, _, _ = ti.macd(data['Adj Close'].values, 12, 26, 9)
    data['MACD'] = [np.nan] * (len(data) - len(ma_cd)) + list(ma_cd)
    lower, _, upper = ti.bbands(data['Adj Close'].values, period=20, stddev=2)
    data['UpperBollingerBand'] = [np.nan] * (len(data) - len(upper)) + list(upper)
    data['LowerBollingerBand'] = [np.nan] * (len(data) - len(lower)) + list(lower)

    technical_indicator_features = ['MA20', 'MA50', 'RSI', 'MACD', 'UpperBollingerBand', 'LowerBollingerBand']

    return data, technical_indicator_features


def train_validation_test_split(df_hist, returns, technical_indicator_features, **kwargs):
    """
    Split data into training and validation test. Note that as
    the data is time series, so, split will not be random. Keep
    1 year of data for validation purpose and last 1 year of data
    for testing purpose and rest all for training.

    Args:
        df_hist (pd.DataFrame): historical complete data
        returns (bool): filter to select either daily returns or closing price of the historial data as a feature
        technical_indicator_features (list): list of technical indicators features added
        **kwargs: training and validation test split

    Returns:
        training and validation dataset
    """
    df_hist['daily_returns'] = df_hist['Adj Close'].pct_change()
    df_hist = df_hist.dropna()
    if returns:
        col = 'daily_returns'
    else:
        col = 'Adj Close'
    # create rolling dataset
    dt_model = create_features_and_target_split(pd.DataFrame(df_hist[col]), col,
                                                kwargs['past_day_returns_for_predicting'])
    dt_model = pd.merge(dt_model, df_hist, how="left", left_index=True, right_index=True)

    return standardize_and_limit_outliers_returns(dt_model, kwargs['past_day_returns_for_predicting'],
                                                  technical_indicator_features, **kwargs)
