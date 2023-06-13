"""
Method that can be used to compute multistep forecasting
Although I decided to not use it in results as the results
were completely off, and it requires more time to build
sophisticated features.
"""
import pandas as pd
import numpy as np

from model.utils.pre_processing import add_technical_indicators, create_features_and_target_split


def get_features_for_multi_step_forecasting(price_history, col, rol_freq):
    """
    This method gets predicted price history. In multi-step forecasting, we use
    predicted data as an input in model to create input features used in model
    and then use those features to make predictions for comparing model.

    Args:
        price_history (numpy array): predicted price history which we will use to compute features
        col (str): features that will be used for prediction, either daily returns or closing prices
        rol_freq (int): number of past days consider for predicting next day return

    Returns:
        numpy array
    """
    price_history.rename(columns={'Predicted Adj Close': 'Adj Close'},
                         inplace=True)  # to leverage the same existing code used in training
    price_history, technical_indicator_features = add_technical_indicators(price_history)
    dt_test_model = create_features_and_target_split(pd.DataFrame(price_history[col]), col, rol_freq)
    dt_test_model = pd.merge(dt_test_model, price_history, how="left", left_index=True, right_index=True)
    feature_ls = np.concatenate(
        (dt_test_model.iloc[-1, :rol_freq].values, dt_test_model.iloc[-1][technical_indicator_features].values))

    return feature_ls
