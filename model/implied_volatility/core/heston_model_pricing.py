"""
This module contains numerical method approach
to determine heston model price for a set of
parameters
"""

import numpy as np
from model.implied_volatility.core.implied_vol import implied_volatility_call


def get_heston_price(row):
    """
    Generates simultaneous simulation of stock price and
    volatility process to solve for final price using numerical method

    Args:
        row (pd.DataFrame) : input parameters for an option
    """
    steps = 2000
    Npaths = 10000
    dt = row['time_to_maturity'] / steps
    size = (Npaths, steps)
    prices = np.zeros(size)
    K = row['strike']
    s_t = row['s0']
    v_t = row['initial_variance']
    rho = row['correlation']
    kappa = row['reversion_speed']  # mean reversion speed
    xi = row['vol_vol']
    theta = row['Long_average_variance']
    r = row['risk_free_rate']
    T = row['time_to_maturity']
    for t in range(steps):
        WT = np.random.multivariate_normal(np.array([0, 0]), cov=np.array([[1, rho], [rho, 1]]),
                                           size=Npaths) * np.sqrt(dt)

        s_t = s_t * (np.exp((r - 0.5 * v_t) * dt + np.sqrt(v_t) * WT[:, 0]))
        v_t = np.abs(v_t + kappa * (theta - v_t) * dt + xi * np.sqrt(v_t) * WT[:, 1])
        prices[:, t] = s_t

    final_state_price = prices[:, -1]
    if row['kind'] == "call":
        opt_price = np.mean(np.maximum(final_state_price - K, 0)) * np.exp(-r * T)
    else:
        opt_price = np.mean(np.maximum(K - final_state_price, 0)) * np.exp(-r * T)

    return opt_price


def create_heston_dataset(df, s0, input_param):
    """
    This method takes current underlying price and
    a set of other parameters and then computes each
    option heston model price using numerical method

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
    input_param['european'] = True if df['opt_type'].iloc[0] == "european" else False
    input_param['kind'] = "call" if df['opt_kind'].iloc[0] == "call" else "put"
    input_param['calender_days'] = input_param.apply(lambda x: x['time_to_maturity'] * 365, axis=1)
    input_param['calender_days'] = input_param['calender_days'].round().astype(int)
    input_param['Heston_price'] = input_param.apply(lambda x: get_heston_price(x), axis=1)
    input_param['implied_vol'] = input_param.apply(lambda x: implied_volatility_call(x), axis=1)
    input_param['opt_price_by_strike'] = input_param.apply(lambda x: x['Heston_price'] / x['strike'], axis=1)

    return input_param
