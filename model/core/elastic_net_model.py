"""
This module build elastic net linear regression
which uses penalties from both lasso and ridge
regression to regularize regression models.
"""
import pickle
import collections
import numpy as np
import pandas as pd

from datetime import date, datetime, timedelta
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import PredefinedSplit, GridSearchCV
from model.utils.pre_processing import train_validation_test_split, add_technical_indicators
from model.utils.get_data import get_historical_stock_data


def elastic_net_hyper_parameter_tuning(features, target, features_val, target_val):
    """
    build and fit the elastic net model and return
    the best model hyper-parameter
    Args:
        features (pd.DataFrame) : lagged daily returns for a rolling window as feature
        target (pd.DataFrame) : next day return as target
        features_val (pd.DataFrame) : validation set features fitting lambda1 and lambda2
        target_val (pd.DataFrame) : validation set target

    Returns:
        model
    """
    elastic_net = ElasticNet()
    # confusingly, alpha parameter in theory is set by l1 ratio in ElasticNet library
    # and lambda by alpha that controls sum of both penalties
    param_grid = {
        'alpha': np.logspace(-5, 2, 8),
        'l1_ratio': [.1, .2, .3, .4, .5, .6, .7, .8, .9]
    }
    # keeping the same holdout validation set for tuning
    split_index = [-1] * len(features) + [0] * len(features_val)
    pds = PredefinedSplit(test_fold=split_index)
    X = np.concatenate((features, features_val), axis=0)
    y = np.concatenate((target, target_val), axis=0)
    # hyper-parameter tuning
    clf = GridSearchCV(elastic_net, cv=pds, param_grid=param_grid, scoring="neg_mean_squared_error", verbose=True)
    clf.fit(X, y)

    return clf.best_params_


def elastic_model(stock, features, target, alpha, l1_ratio):
    """

    Args:
        stock (str): stock name
        features (pd.DataFrame): lagged daily returns for a rolling window as feature
        target (pd.DataFrame): next day return as target
        alpha (float): lambda parameter
        l1_ratio (float): ratio for individual penalty

    Returns:
        save the model for stock
    """
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
    model.fit(features, target)

    # save the model
    model_save_path = r"./model/output/ElasticNet/" + stock + ".pkl"
    with open(model_save_path, 'wb') as f:
        pickle.dump(model, f)

    return model


def run_elastic_net_model_for_all_stocks():
    """
    This method first reads all the stock tickers,
    retrieve data, creates training set, hold out
    validation set and testing in real world data.
    Find the best hyperparameters for each stock and
    then build and save model.

    Returns:
        save all the models, dictionary with model details
    """
    start_date = date(2016, 1, 1)
    end_date = datetime.today()
    data_dict = get_historical_stock_data(start_date, end_date)
    # training data ends 2 years before while validation data starts after that and ends 1 year before
    param = {'training_end': end_date - timedelta(seconds=2 * 365.2425 * 24 * 60 * 60),
             'validation_end': end_date - timedelta(seconds=1 * 365.2425 * 24 * 60 * 60),
             'past_day_returns_for_predicting': 21}
    # running for all stocks
    rol_freq = param['past_day_returns_for_predicting']
    model_details = collections.OrderedDict()
    for key, data in data_dict.items():
        data = add_technical_indicators(data)
        X_train_norm, Y_train, X_val_norm, Y_val, scaler = train_validation_test_split(data, **param)
        best_hyper_parameter = elastic_net_hyper_parameter_tuning(X_train_norm, Y_train, X_val_norm, Y_val)
        final_model = elastic_model(key, X_train_norm, Y_train, best_hyper_parameter['alpha'],
                                    best_hyper_parameter['l1_ratio'])
        model_details[key] = {
            'model': final_model,
            'scaler': scaler
        }

    return model_details
