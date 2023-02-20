"""
This module runs the main neural network model
and also tune the hyperparameter, store them to
a output file and test accuracy
"""

import json
import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from scikeras.wrappers import KerasRegressor
from model.pricing.utils.input import read_hyper_parameters_range


def find_best_hyper_parameter_config(param_grid, dt_set):
    """
    This method creates the neural network
    architecture for a given set of hyperparameter

    Args:
        param_grid (dict) : multiple choices for hyperparameter selection
        dt_set (pd.DataFrame) : full dataset

    Returns:
        pd.DataFrame
    """
    feature_columns = ['moneyness', 'time_to_maturity', 'risk_free_rate', 'volatility']
    input_features = dt_set[feature_columns]
    target = dt_set['opt_price_by_strike']
    x_train, x_test, y_train, y_test = train_test_split(input_features, target, test_size=0.2, random_state=11)
    x_train = np.float32(x_train)
    y_train = np.float32(y_train)

    def create_model(neurons=200, activation="relu", initialization="uniform", batch_normalisation="yes",
                     optimizer="SGD", drop_out_rate=0.0):
        """Define model and test it"""
        model_ind = tf.keras.Sequential()
        model_ind.add(tf.keras.layers.Dense(neurons, activation=activation, kernel_initializer=initialization,
                                            input_shape=(input_features.shape[1],)))
        if batch_normalisation == "yes":
            model_ind.add(tf.keras.layers.BatchNormalization())
        model_ind.add(tf.keras.layers.Dropout(drop_out_rate))
        model_ind.add(tf.keras.layers.Dense(neurons, activation=activation, kernel_initializer=initialization))
        if batch_normalisation == "yes":
            model_ind.add(tf.keras.layers.BatchNormalization())
        model_ind.add(tf.keras.layers.Dropout(drop_out_rate))
        model_ind.add(tf.keras.layers.Dense(neurons, activation=activation, kernel_initializer=initialization))
        if batch_normalisation == "yes":
            model_ind.add(tf.keras.layers.BatchNormalization())
        model_ind.add(tf.keras.layers.Dropout(drop_out_rate))
        model_ind.add(tf.keras.layers.Dense(1))
        model_ind.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mse'])

        return model_ind

    k_model = KerasRegressor(model=create_model, verbose=0, neuron=None, activation=None, initialization=None,
                             batch_normalisation=None, drop_out_rate=None)
    grid = RandomizedSearchCV(estimator=k_model, cv=KFold(3), param_distributions=param_grid, verbose=1, n_iter=3,
                              n_jobs=-1, scoring="neg_mean_squared_error", error_score="raise")
    grid_results = grid.fit(x_train, y_train, epochs=100, verbose=0)

    return grid_results.best_params_


def create_set_of_hyperparameter():
    """
    This method first reads the hyperparameter range
    and then creates a dataframe of hyperparameter

    Returns:
         pd.DataFrame
    """
    hyper_parameters = read_hyper_parameters_range()
    param_grid = {
        'activation': hyper_parameters['activation'],
        'neurons': list(np.arange(hyper_parameters['neurons'][0], hyper_parameters['neurons'][1], 100)),
        'drop_out_rate': hyper_parameters['dropout_rate'],
        'initialization': hyper_parameters['initialization'],
        'batch_normalisation': hyper_parameters['batch_normalisation'],
        'optimizer': hyper_parameters['optimizer'],
        'batch_size': list(np.arange(hyper_parameters['batch_size'][0], hyper_parameters['batch_size'][1], 256))
    }

    return param_grid


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def hyperparameter_tuning(dt_set):
    """
    This method split data set into train, test and then
    creates model, finds the best possible hyperparameter,
    stores them to a file and then predict its accuracy on
    the test set.

    Loss function is the one where the model is trained but
    you can always define multiple metrics where you want to
    track the performance of the model

    Args:
        dt_set (pd.DataFrame) : dataset
    """
    param_grid = create_set_of_hyperparameter()
    best_hyper_param = find_best_hyper_parameter_config(param_grid, dt_set)
    pt = r"./model/output/"
    file_name = pt + "best_hyper_parameter.json"
    with open(file_name, 'w') as fp:
        json.dump(best_hyper_param, fp, cls=NpEncoder)
