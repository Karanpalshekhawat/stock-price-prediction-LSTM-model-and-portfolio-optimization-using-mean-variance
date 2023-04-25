"""
This module transforms historical data set
of stock price evolutions into structured dataset
of features and target that can be used to build model for LSTM.
I have kept it separate from Elastic net as one major difference
is it is that I am using directly closing prices to predict
next day closing prices instead of dealing with returns, and I can also
add separate extra features in this if needed.
"""
import pandas as pd
import numpy as np
import tulipy as ti

from sklearn.preprocessing import StandardScaler


def create_features_and_target_split_lstm(df, rol_freq):
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
        features_ls = df.iloc[i - rol_freq:i, 0:df.shape[1]]['Adj Close'].to_list()
        target_ls = df.iloc[i:i + 1, 0:df.shape[1]]['Adj Close'].to_list()
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
    rsi = ti.rsi(data['Adj Close'].values, period=14)
    data['RSI'] = [np.nan] * (len(data) - len(rsi)) + list(rsi)
    ma_cd, _, _ = ti.macd(data['Adj Close'].values, 12, 26, 9)
    data['MACD'] = [np.nan] * (len(data) - len(ma_cd)) + list(ma_cd)
    lower, _, upper = ti.bbands(data['Adj Close'].values, period=20, stddev=2)
    data['UpperBollingerBand'] = [np.nan] * (len(data) - len(upper)) + list(upper)
    data['LowerBollingerBand'] = [np.nan] * (len(data) - len(lower)) + list(lower)

    return data


def train_validation_test_split(df_hist, **kwargs):
    """
    Split data into training and validation test. Note that as
    the data is time series, so, split will not be random. Keep
    1 year of data for validation purpose and last 1 year of data
    for testing purpose and rest all for training.

    Args:
        df_hist (pd.DataFrame): historical complete data
        **kwargs: training and validation test split

    Returns:
        training and validation dataset
    """
    df_hist['daily_returns'] = df_hist['Adj Close'].pct_change()
    df_hist = df_hist.dropna()
    # create rolling dataset
    dt_model = create_features_and_target_split_lstm(pd.DataFrame(df_hist['Adj Close']),
                                                kwargs['past_day_returns_for_predicting'])
    dt_model = pd.merge(dt_model, df_hist, how="left", left_index=True, right_index=True)
    technical_indicator_features = ['RSI', 'MACD', 'UpperBollingerBand', 'LowerBollingerBand']
    # normalize and remove outliers

    return standardize_and_limit_outliers_returns(dt_model, kwargs['past_day_returns_for_predicting'],
                                                  technical_indicator_features, **kwargs)


def get_features_for_multi_step_forecasting(price_history, rol_freq, technical_indicator_features):
    """
    This method gets predicted price history. In multi-step forecasting, we use
    predicted data as an input in model to create input features used in model
    and then use those features to make predictions for comparing model.

    Args:
        price_history (numpy array): predicted price history which we will use to compute features
        rol_freq (int): number of past days consider for predicting next day return
        technical_indicator_features (list) : feature columns for technical indicators

    Returns:
        numpy array
    """
    price_history.rename(columns={'Predicted Adj Close': 'Adj Close'},
                         inplace=True)  # to leverage the same existing code used in training
    price_history = add_technical_indicators(price_history)
    price_history['daily_returns'] = price_history['Adj Close'].pct_change()
    dt_test_model = create_features_and_target_split_lstm(pd.DataFrame(price_history['daily_returns']), rol_freq)
    dt_test_model = pd.merge(dt_test_model, price_history, how="left", left_index=True, right_index=True)
    feature_ls = np.concatenate(
        (dt_test_model.iloc[-1, :rol_freq].values, dt_test_model.iloc[-1][technical_indicator_features].values))

    return feature_ls
