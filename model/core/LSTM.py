"""
This module build LSTM model for a set of
hyper parameters and defined structured layers.
Features creation and other data pre-processing
steps are similar to ElasticNet model and
is leveraged from there.
"""

from keras.layers.core import Dense, Dropout
from keras.layers import LSTM
from keras.models import Sequential
from keras.optimizers import Adam


def build_lstm_model_structure(num_features, num_units, activation='tanh', dropout_rate=0.2, learning_rate=0.001):
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
    model.add(LSTM(units=num_units, activation=activation, return_sequences=True, input_shape=num_features))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(units=num_units, activation=activation, return_sequences=True))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(units=num_units, activation=activation))
    # Add a dense output layer with a single output unit
    model.add(Dense(units=1))

    # Define optimizer
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error')

    return model


def run_lstm_model_for_all_stocks():
    """
    This method first reads all the stock tickers,
    retrieve data, creates training set, hold out
    validation set and then train the LSTM model.

    Returns:
        save all the models, dictionary with model details
    """

    return
