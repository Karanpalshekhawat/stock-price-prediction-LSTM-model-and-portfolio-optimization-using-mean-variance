"""
This module contains method to read the trained model,
make prediction using it and then plotting the chart
as per requirement.
"""

import pickle
import datetime as dt
import pandas as pd
import numpy as np
import yfinance as yf

from pandas.tseries.offsets import BDay

from model.utils.pre_processing import create_features_and_target_split, standardize_and_limit_outliers_returns, \
    add_technical_indicators
from model.utils.pre_processing_LSTM import get_features_for_multi_step_forecasting


def read_trained_model_and_scaler(stock, path):
    """
    It reads the trained model which is stored in the pkl format

    Args:
        stock (str): stock name
        path (str): path to read it from

    Returns:
        model, scaler
    """
    model_save_path = path + stock + ".pkl"
    with open(model_save_path, 'rb') as f:
        model = pickle.load(f)

    scaler_save_path = path + r"/scaler/" + stock + ".pkl"
    with open(scaler_save_path, 'rb') as f:
        scaler = pickle.load(f)

    return model, scaler


def get_data(ticker, start_date, end_date):
    """
    This module retrieves data from yahoo finance for plotting.

    Args:
        ticker (str): Stock ticker to download data from yf
        start_date (datetime) : starting date of data history
        end_date (datetime) : end date till data is retrieved

    Returns:
        pd.DataFrame
    """
    df_hist = yf.download(ticker, start=start_date, end=end_date)
    df_all_dates = pd.DataFrame(index=pd.bdate_range(start=start_date, end=end_date))
    df_hist = pd.merge(df_all_dates, df_hist, how="left", left_index=True, right_index=True)
    df_hist = df_hist.fillna(method='ffill')

    return df_hist


def pre_processing(df_hist, rol_freq):
    """
    This method performs all pre-processing steps
    to create features and align data that can be used for further analysis

    Args:
        df_hist (pd.DataFrame): historical dataset of the stock
        rol_freq (int): number of past days used as features in training the model

    Returns:
        pd.DataFrame
    """
    df_hist, technical_indicator_features = add_technical_indicators(df_hist)
    df_hist['daily_returns'] = df_hist['Adj Close'].pct_change()
    df_hist = df_hist.dropna()
    col = 'Adj Close'
    dt_model = create_features_and_target_split(pd.DataFrame(df_hist[col]), col, rol_freq)
    dt_model = pd.merge(dt_model, df_hist, how="left", left_index=True, right_index=True)

    return dt_model, technical_indicator_features


def benchmark_indexed(df):
    """
    This method is used to nicely compare the actual trend
    with the predicted trend to be benchmarked as evolution of
    INR 100.

    Args:
        df(pd.DataFrame): predicted value, actual value and features dataframe

    Returns:
        pd.DataFrame
    """
    df['Actual'] = ""
    df['Predicted'] = ""
    df.iloc[1, df.columns.get_loc('Actual')] = 100
    df.iloc[1, df.columns.get_loc('Predicted')] = 100

    for i in range(2, len(df)):
        start = df.iloc[i - 1, df.columns.get_loc('Predicted_close')]
        end = df.iloc[i, df.columns.get_loc('Predicted_close')]
        predicted_return = (end / start) - 1
        df.iloc[i, df.columns.get_loc('predicted_returns')] = predicted_return
        df.iloc[i, df.columns.get_loc('Predicted')] = (1 + predicted_return) * df.iloc[
            i - 1, df.columns.get_loc('Predicted')]
        # actual trend
        actual_return = df.iloc[i, df.columns.get_loc('actual_returns')]
        df.iloc[i, df.columns.get_loc('Actual')] = (1 + actual_return) * df.iloc[i - 1, df.columns.get_loc('Actual')]

    return df


def forecast_one_day(df, rol_freq, model, scaler, technical_indicator_features):
    """
    This method restructures input data to predict next day price.
    Note that inputs are the adjusted on a daily basis with actual realized
    stock price.

    Args:
        df: features in dataframe
        rol_freq (int): number of past days used as features in training
        model: Trained model
        scaler: Trained scaler which is used to normalize the features
        technical_indicator_features (list): list of technical features used in model training

    Returns:
        df
    """
    df['Predicted_close'] = ""
    df['actual_returns'] = df['Adj Close'].pct_change()
    df = df.dropna()
    df['predicted_returns'] = ""
    for i in range(0, len(df) - 1):
        features_ls = np.concatenate((df.iloc[i, :rol_freq].values, df.iloc[i][technical_indicator_features].values))
        features_ls = features_ls.reshape(-1, 1)
        new_scaled_ls = scaler.transform(features_ls.T)
        new_scaled_ls = np.asarray(new_scaled_ls).astype(np.float32)
        new_scaled_ls = new_scaled_ls.reshape((new_scaled_ls.shape[0], new_scaled_ls.shape[1], 1))
        predicted_value = model.predict(new_scaled_ls, verbose=0)[0][0]
        # current features are predicting next day return
        df.iloc[i + 1, df.columns.get_loc('Predicted_close')] = predicted_value

    df = benchmark_indexed(df)

    return df


def multistep_forecasting(df_multi_step, rol_freq, model, scaler, num_days):
    """
    This method performs multi step forecasting i.e. it restructures input data
    to predict next day price and then uses predicted price as inputs to predict
    the further day stock price and so on. Only argument  extra to this method is
    num_days which is the number of days in future you want to predict.

    Returns:
        pd.DataFrame
    """
    df_multi_step['Predicted Adj Close'] = df_multi_step['Adj Close']  # will update it while looping
    end_date = dt.datetime.today()
    period_start_date = end_date - dt.timedelta(days=num_days)
    predicted_dates = pd.bdate_range(start=period_start_date, end=end_date)
    col = "Adj Close"
    for date in predicted_dates[:-1]:
        price_trend_before_date = pd.DataFrame(
            df_multi_step.loc[df_multi_step.index < date, 'Predicted Adj Close'])
        df = price_trend_before_date.copy()
        features_ls = get_features_for_multi_step_forecasting(df, col, rol_freq)
        features_ls = features_ls.reshape(-1, 1)
        new_scaled_ls = scaler.transform(features_ls.T)
        new_scaled_ls = np.asarray(new_scaled_ls).astype(np.float32)
        new_scaled_ls = new_scaled_ls.reshape((new_scaled_ls.shape[0], new_scaled_ls.shape[1], 1))
        predicted_value = model.predict(new_scaled_ls, verbose=0)[0][0]
        next_working_date = (date + BDay(1)).replace(hour=0, minute=0, second=0, microsecond=0)
        date = date.replace(hour=0, minute=0, second=0, microsecond=0)
        df_multi_step.loc[next_working_date, 'Predicted Adj Close'] = predicted_value

    return df_multi_step, predicted_dates[0]
