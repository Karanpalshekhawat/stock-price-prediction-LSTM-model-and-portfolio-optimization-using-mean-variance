"""
This module build LSTM model for a set of
hyper parameters and defined structured layers.
Features creation and other data pre-processing
steps are similar to ElasticNet model and
is leveraged from there.
"""
import json
import pickle
import collections
import itertools
import numpy as np
from datetime import datetime, timedelta

from keras.layers.core import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.models import Sequential
from keras.optimizers import Adam, RMSprop
from tensorflow_addons.metrics import RSquare

from sklearn.model_selection import PredefinedSplit, GridSearchCV
from model.utils.pre_processing_LSTM import train_validation_test_split, add_technical_indicators


def create_lstm_model(num_features, neurons=100, activation='relu', dropout_rate=0.2, learning_rate=0.0001,
                      optimizer="RMSprop", initialization="he_uniform"):
    """
    This method defines the layered structure
    of a LSTM model for a set of parameters.

    Args:
        num_features (int): number of features used to predict next day return
        neurons (int): number of units in each LSTM layer, number of layers are fixed
        activation (string): activation function for the LSTM layers, e.g. 'tanh', 'relu', 'sigmoid'
        dropout_rate (float): Drop out rate after each layer for tackling over-fitting problem
        learning_rate (float): learning rate for the Adam optimizer
        optimizer (str): optimizer choosing conditions
        initialization (str): initialisation weight parameters

    Returns:
        LSTM model
    """
    model = Sequential()

    # Add 3 LSTM layers with the same number of units and activation function
    model.add(LSTM(units=neurons, activation=activation, return_sequences=True, kernel_initializer=initialization,
                   input_shape=(num_features, 1)))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(units=neurons, activation=activation, return_sequences=True, kernel_initializer=initialization))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(units=neurons, activation=activation))
    # Add a dense output layer with a single output unit
    model.add(Dense(units=1))
    model.add(Activation('linear'))

    # Define optimizer
    if optimizer == "RMSprop":
        optimizer = RMSprop(learning_rate=learning_rate)
    else:
        optimizer = Adam(learning_rate=learning_rate)

    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=[RSquare()])

    return model


def tune_hyper_parameter(param_grid, X_train, Y_train, X_val, Y_val):
    """
    This method creates the neural network
    architecture for a given set of hyperparameter

    Args:
        param_grid (dict) : multiple choices for hyperparameter selection
        X_train (np.array) : training dataset
        Y_train (np.array) : target values in training dataset
        X_val (np.array) : validation dataset
        Y_val (np.array) : target values in validation dataset

    Returns:
        best hyperparams
    """
    model = create_lstm_model(num_features=X_train.shape[1])

    # implemented own defined grid search as I was facing many issues using Grid search CV
    # to match sk-learn method with keras method hyper parameters definitions
    split_index = [-1] * len(X_train) + [0] * len(X_val)
    pds = PredefinedSplit(test_fold=split_index)
    X = np.concatenate((X_train, X_val), axis=0)
    y = np.concatenate((Y_train, Y_val), axis=0)

    clf = GridSearchCV(estimator=model, cv=pds, param_grid=param_grid, scoring="accuracy", verbose=True)
    clf.fit(X, y)

    print('Best hyperparameters: ', clf.best_params_)
    print('Validation accuracy: ', clf.best_score_)

    # Evaluate the best model on the validation set
    best_model = clf.best_estimator_
    val_loss, val_acc = best_model.evaluate(X_val, Y_val)

    return


def create_set_of_hyperparameter():
    """
    This method first reads the hyperparameter range
    and then creates a dataframe of hyperparameter

    Returns:
         pd.DataFrame
    """
    pt = r"./model/static_data/"
    json_file = pt + "hyper-parameters-range.json"
    with open(json_file) as f:
        hyper_parameters = json.load(f)
    param_grid = {
        'activation': hyper_parameters['activation'],
        'neurons': list(np.arange(hyper_parameters['neurons'][0], hyper_parameters['neurons'][1], 50)),
        'initialization': hyper_parameters['initialization'],
        'optimizer': hyper_parameters['optimizer'],
        'batch_size': hyper_parameters['batch_size'],
        'learning_rate': hyper_parameters['learning_rate'],
        'epochs': hyper_parameters['epochs']
    }
    combinations = list(itertools.product(*param_grid.values()))
    final_params_dict = collections.OrderedDict()
    for i, ls in enumerate(combinations):
        final_params_dict[i] = {'activation': ls[0], 'neurons': ls[1], 'initialization': ls[2], 'optimizer': ls[3],
                                'batch_size': ls[4], 'learning_rate': ls[5], 'epochs': ls[6]}

    return final_params_dict


def restructure_data(X_train, Y_train, X_val, Y_val):
    """
    This method just restructure dataset for appropriate type.
    """
    # restructure training data
    X_train = np.asarray(X_train).astype(np.float32)
    Y_train = np.asarray(Y_train).astype(np.float32)
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    Y_train = Y_train.reshape(Y_train.shape[0], 1)

    # restructure validation data
    X_val = np.asarray(X_val).astype(np.float32)
    Y_val = np.asarray(Y_val).astype(np.float32)
    X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))
    Y_val = Y_val.reshape(Y_val.shape[0], 1)

    return X_train, Y_train, X_val, Y_val


def run_lstm_model_for_all_stocks(data_dict, end_date):
    """
    This method first reads all the stock tickers,
    retrieve data, creates training set, hold out
    validation set and then train the LSTM model.

    Args:
        data_dict (dict) : historical dataset for all stocks consider in analysis
        end_date (datetime) : end period date

    Returns:
        save all the models, dictionary with model details
    """
    param = {'training_end': end_date - timedelta(seconds=2 * 365.2425 * 24 * 60 * 60),
             'validation_end': end_date - timedelta(seconds=1 * 365.2425 * 24 * 60 * 60),
             'past_day_returns_for_predicting': 10}
    # running for all stocks
    model_details = collections.OrderedDict()
    for key, data in data_dict.items():
        data = add_technical_indicators(data)
        X_train, Y_train, X_val, Y_val, scaler = train_validation_test_split(data, **param)
        X_train, Y_train, X_val, Y_val = restructure_data(X_train, Y_train, X_val, Y_val)

        lstm_model = create_lstm_model(num_features=new_x_tensor.shape[1], neurons=50)
        lstm_model.fit(new_x_tensor, new_y_tensor, batch_size=100, epochs=300, verbose=True)
        model_details[key] = {
            'model': lstm_model,
            'scaler': scaler
        }
        # save the model
        model_save_path = r"./model/output/LSTM/" + key + ".pkl"
        with open(model_save_path, 'wb') as f:
            pickle.dump(lstm_model, f)

    return
