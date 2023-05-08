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

from model.utils.pre_processing_LSTM import train_validation_test_split, add_technical_indicators


def create_lstm_model(num_features, dropout_rate=0.2, **kwargs):
    """
    This method defines the layered structure
    of a LSTM model for a set of parameters.

    Args:
        num_features (int): number of features used to predict next day return
        dropout_rate (float): fixed dropout rate

    Returns:
        LSTM model
    """
    model = Sequential()

    # Add 3 LSTM layers with the same number of units and activation function
    model.add(LSTM(units=kwargs['neurons'], activation=kwargs['activation'], return_sequences=True,
                   kernel_initializer=kwargs['initialization'], input_shape=(num_features, 1)))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(units=kwargs['neurons'], activation=kwargs['activation'], return_sequences=True,
                   kernel_initializer=kwargs['initialization']))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(units=kwargs['neurons'], activation=kwargs['activation']))
    # Add a dense output layer with a single output unit
    model.add(Dense(units=1))
    model.add(Activation('linear'))

    # Define optimizer
    if kwargs['optimizer'] == "RMSprop":
        optimizer = RMSprop(learning_rate=kwargs['learning_rate'])
    else:
        optimizer = Adam(learning_rate=kwargs['learning_rate'])

    model.compile(optimizer=optimizer, loss='mean_squared_error')

    return model


def tune_hyper_parameter(final_params_dict, X_train, Y_train, X_val, Y_val):
    """
    This method creates the neural network
    architecture for a given set of hyperparameter

    Args:
        final_params_dict (dict) : multiple choices for hyperparameter selection
        X_train (np.array) : training dataset
        Y_train (np.array) : target values in training dataset
        X_val (np.array) : validation dataset
        Y_val (np.array) : target values in validation dataset

    Returns:
        best hyperparams
    """
    # implemented self defined grid search as I was facing many issues using Grid search CV
    # with match sk-learn compatible hyper-parameters with keras method hyper-parameters definitions
    loss_score = collections.OrderedDict()
    for key, sub_dict in final_params_dict.items():
        print(f"Model Number : {key}")
        model = create_lstm_model(num_features=X_train.shape[1], **sub_dict)
        model.fit(X_train, Y_train, batch_size=sub_dict['batch_size'], epochs=sub_dict['epochs'], verbose=True)
        loss_score[key] = model.evaluate(X_val, Y_val)

    lowest_score_key = min(loss_score, key=lambda k: loss_score[k])

    return final_params_dict[lowest_score_key]


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
             'past_day_returns_for_predicting': 21}
    # running for all stocks
    hyper_params_dict = create_set_of_hyperparameter()
    model_details = collections.OrderedDict()
    for key, data in data_dict.items():
        data = add_technical_indicators(data)
        X_train, Y_train, X_val, Y_val, scaler = train_validation_test_split(data, **param)
        X_train, Y_train, X_val, Y_val = restructure_data(X_train, Y_train, X_val, Y_val)
        param_dict = tune_hyper_parameter(hyper_params_dict, X_train, Y_train, X_val, Y_val)
        # save the model
        lstm_model = create_lstm_model(num_features=X_train.shape[1], **param_dict)
        lstm_model.fit(X_train, Y_train, batch_size=param_dict['batch_size'], epochs=param_dict['epochs'], verbose=True)
        model_save_path = r"./model/output/LSTM/" + key + ".pkl"
        with open(model_save_path, 'wb') as f:
            pickle.dump(lstm_model, f)

        model_details[key] = {
            'model': lstm_model,
            'scaler': scaler
        }

    return model_details
