"""
This module build LSTM model for a set of
hyper parameters and defined structured layers.
Features creation and other data pre-processing
steps are similar to ElasticNet model and
is leveraged from there.
"""
import pickle
import collections
import numpy as np
from datetime import datetime, timedelta

from keras.layers.core import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.models import Sequential
from keras.optimizers import Adam, RMSprop
from tensorflow_addons.metrics import RSquare

from model.utils.pre_processing_LSTM import train_validation_test_split, add_technical_indicators


def build_lstm_model_structure(num_features, num_units, activation='relu', dropout_rate=0.2, learning_rate=0.0001):
    """
    This method defines the layered structure
    of a LSTM model for a set of parameters.

    Args:
        num_features (int): number of features used to predict next day return
        num_units (int): number of units in each LSTM layer, number of layers are fixed
        activation (string): activation function for the LSTM layers, e.g. 'tanh', 'relu', 'sigmoid'
        dropout_rate (float): Drop out rate after each layer for tackling over-fitting problem
        learning_rate (float): learning rate for the Adam optimizer

    Returns:
        LSTM model
    """
    model = Sequential()

    # Add 3 LSTM layers with the same number of units and activation function
    model.add(LSTM(units=num_units, activation=activation, return_sequences=True, input_shape=(num_features, 1)))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(units=num_units, activation=activation, return_sequences=True))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(units=num_units, activation=activation))
    # Add a dense output layer with a single output unit
    model.add(Dense(units=1))
    model.add(Activation('linear'))

    # Define optimizer
    optimizer = RMSprop(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=[RSquare()])

    return model


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
        X_train_norm, Y_train, X_val_norm, Y_val, scaler = train_validation_test_split(data, **param)
        X_train_norm = np.asarray(X_train_norm).astype(np.float32)
        Y_train = np.asarray(Y_train).astype(np.float32)
        new_x_tensor = X_train_norm.reshape((X_train_norm.shape[0], X_train_norm.shape[1], 1))
        new_y_tensor = Y_train.reshape(Y_train.shape[0], 1)
        lstm_model = build_lstm_model_structure(num_features=new_x_tensor.shape[1], num_units=50)
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
