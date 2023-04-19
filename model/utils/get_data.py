"""
This module contains method which is used to extract
historical data for NSE listed stocks for a given
start and end date. Relevant tickers are stored in
static data yaml file.
"""

import yaml
import collections
import pandas as pd
import yfinance as yf


def read_yf_tickers():
    """
    Reading static data file to get
    the ticker for downloading historical data
    """
    pt = r"./model/static_data/"
    yml_file = pt + "list_of_stocks.yml"
    with open(yml_file) as file:
        df = pd.json_normalize(yaml.safe_load(file))
    return df


def get_historical_stock_data(start_date, end_date):
    """
    Get historical stock data for a set of tickers

    Args:
        start_date: Starting date of a period
        end_date: Ending date of a period

    Returns:
        dict
    """
    df_tickers = read_yf_tickers()
    otpt_dict = collections.OrderedDict()
    for ticker, stock in zip(df_tickers['ticker'].values, df_tickers['stock'].values):
        df_hist = yf.download(ticker, start=start_date, end=end_date)
        df_all_dates = pd.DataFrame(index=pd.bdate_range(start=start_date, end=end_date))
        df_hist = pd.merge(df_all_dates, df_hist, how="left", left_index=True, right_index=True)
        df_hist = df_hist.fillna((df_hist.shift() + df_hist.shift(-1)) / 2)
        otpt_dict[stock] = df_hist

    return otpt_dict
