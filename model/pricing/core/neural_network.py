"""
This module contains the model definition
and a method to identify the best learning rate
"""

import tensorflow as tf

from sklearn.model_selection import train_test_split
from tensorflow_addons.optimizers import CyclicalLearningRate


def nn_model(row, features):
    """
    Just defines the model structure

    Args:
        row (pd.DataFrame) : hyperparameters
        features (int): number of features

    Return:
        Sequential
    """
    model_ind = tf.keras.Sequential()
    model_ind.add(tf.keras.layers.Dense(row['neurons'], activation=row['activation'],
                                        kernel_initializer=row['initialization'], input_shape=(features,)))
    if row['batch_normalisation'] == "yes":
        model_ind.add(tf.keras.layers.BatchNormalization())
    model_ind.add(tf.keras.layers.Dropout(row['drop_out_rate']))

    model_ind.add(tf.keras.layers.Dense(row['neurons'], activation=row['activation'],
                                        kernel_initializer=row['initialization']))
    if row['batch_normalisation'] == "yes":
        model_ind.add(tf.keras.layers.BatchNormalization())
    model_ind.add(tf.keras.layers.Dropout(row['drop_out_rate']))

    model_ind.add(tf.keras.layers.Dense(row['neurons'], activation=row['activation'],
                                        kernel_initializer=row['initialization']))
    if row['batch_normalisation'] == "yes":
        model_ind.add(tf.keras.layers.BatchNormalization())
    model_ind.add(tf.keras.layers.Dropout(row['drop_out_rate']))

    model_ind.add(tf.keras.layers.Dense(1))

    return model_ind


def run_nn_model(dt_set, hyper_param, feature_columns, target):
    """
    This method runs the main model by
    using cyclical learning rate method
    for optimizing learning rate and then
    save it in the output folder for reuse

    Args:
         dt_set (pd.DataFrame) : dataset
         hyper_param (pd.DataFrame) : hyper parameters values
         feature_columns (list) : input features columns
         target (str): target column
    """
    input_features = dt_set[feature_columns]
    target = dt_set[target]
    x_train, x_test, y_train, y_test = train_test_split(input_features, target, test_size=0.2, random_state=11)
    model = nn_model(hyper_param, len(feature_columns))
    steps_per_epoch = len(x_train) // hyper_param['batch_size']
    # you can learn another model to identify the range of learning rate by plotting MSE against different learning rate
    clr = CyclicalLearningRate(initial_learning_rate=1e-5, maximal_learning_rate=1e-3, step_size=2 * steps_per_epoch,
                               scale_fn=lambda x: 1 / (2.0 ** (x - 1)), scale_mode='cycle')
    if hyper_param['optimizer'] == 'SGD':
        optimizer = tf.keras.optimizers.SGD(clr)
    elif hyper_param['optimizer'] == 'Adam':
        optimizer = tf.keras.optimizers.Adam(clr)
    else:
        optimizer = tf.keras.optimizers.RMSprop(clr)
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mse', 'mae'])
    history = model.fit(x_train, y_train, batch_size=hyper_param['batch_size'], validation_data=(x_test, y_test),
                        epochs=30, verbose=0)

    return model, history
