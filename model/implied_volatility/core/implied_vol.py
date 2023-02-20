"""
This module contains brent method to
compute implied volatility of an option
give heston price and other parameters

Advantage of using brent method is that
it's derivative free. Newton raphson method
fails as vega is close to 0 for Deep OTM or
ITM options

It also contains the method to create dataset
that is used to build NN model for computing
implied volatility
"""

import pandas as pd
import numpy as np

from scipy.optimize import brentq
from model.pricing.core.black_scholes import get_bs_price


def implied_volatility_call(row):
    """
    This method compute implied volatility
    using brent method which is derivative free

    Args:
        row (pd.DataFrame) : input parameter

    Returns:
        implied volatility in percent
    """

    def difference(sigma):
        row['volatility'] = sigma
        return row['Heston_price'] - get_bs_price(row)

    try:
        imp_vol = brentq(difference, 0.001, 10, maxiter=1000)
    except:
        imp_vol = row['initial_variance']

    return imp_vol


def get_time_value(row):
    """
    Returns time values of an option

    Args:
        row (pd.DataFrame): pandas series with parameters
    """
    time_value = row['BS_price'] - np.maximum(
        (row['s0'] - row['strike'] * np.exp(-row['time_to_maturity'] * row['risk_free_rate'])), 0)
    return time_value


def create_implied_vol_dataset(df, s0, input_param):
    """
    This method takes current underlying price and
    a set of other parameters and then computes each
    implied vol using numerical method for a given
    BS price

    One extra parameter is used i.e. scaled time value,
    justification for using it is given in detailed in
    literature document

    Args:
        df (pd.DataFrame) : input variables dataframe
        s0 (float) : current underlying price
        input_param (pd.DataFrame) : contains info about money-ness,
                                    time, rfr, volatility to create dataset

    Returns:
         pd.DataFrame
    """
    input_param['s0'] = s0
    input_param['strike'] = input_param.apply(lambda x: x['s0'] / x['moneyness'], axis=1)
    input_param['calender_days'] = input_param.apply(lambda x: x['time_to_maturity'] * 365, axis=1)
    input_param['calender_days'] = input_param['calender_days'].round().astype(int)
    input_param['european'] = True if df['opt_type'].iloc[0] == "european" else False
    input_param['kind'] = "call" if df['opt_kind'].iloc[0] == "call" else "put"
    input_param['BS_price'] = input_param.apply(lambda x: get_bs_price(x), axis=1)
    input_param['time_value'] = input_param.apply(lambda x: get_time_value(x), axis=1)
    input_param = input_param[(input_param['BS_price'] > 0.005) & (input_param['time_value'] > 0.005)]
    input_param['scaled_time_value'] = input_param.apply(lambda x: np.log(x['time_value'] / x['strike']), axis=1)

    return input_param
