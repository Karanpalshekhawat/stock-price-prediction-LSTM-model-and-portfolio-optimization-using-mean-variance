"""
This module contains methods that
create an option object for a set of
parameters and then compute its BS price
"""

from optionprice import Option


def get_bs_price(row):
    """
    This method computes BS price for given
    set of input parameters

    Args:
         row (pd.Series): row series for input df

    Returns:
        float
    """
    opt_obj = Option(
        european=row['european'], kind=row['kind'], s0=row['s0'], k=row['strike'],
        t=row['calender_days'], sigma=row['volatility'], r=row['risk_free_rate']
    )
    return round(opt_obj.getPrice(), 4)


def create_dataset(df, s0, input_param):
    """
    This method takes current underlying price and
    a set of other parameters and then computes each
    option BS price

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
    input_param = input_param[input_param['BS_price'] > 0.01]
    input_param['opt_price_by_strike'] = input_param.apply(lambda x: x['BS_price'] / x['strike'], axis=1)

    return input_param
