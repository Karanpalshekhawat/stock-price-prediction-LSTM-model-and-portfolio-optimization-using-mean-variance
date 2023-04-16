"""
This module transforms historical data set
of stock price evolutions into structured dataset
of features and target that can be used to build model.
"""
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler


def create_features_and_target_split(df, rol_freq):
    """
    It takes input dataframe and creates rolling features
    dataset with specified frequency and also creates
    target which is the next day values.

    Args:
        df (pd.DataFrame):
        rol_freq (int): number of days consider to predict target

    Returns:
        pd.DataFrame
    """
    ls = list(range(1, rol_freq + 1)) + ['target']
    dt_preprocess = pd.DataFrame(columns=ls, index=df.index)
    for i in range(rol_freq, len(df)):
        features_ls = df.iloc[i - rol_freq:i, 0:df.shape[1]]['daily_returns'].to_list()
        target_ls = df.iloc[i:i + 1, 0:df.shape[1]]['daily_returns'].to_list()
        dt_preprocess.iloc[i] = features_ls + target_ls

    dt_preprocess = dt_preprocess.dropna()

    return dt_preprocess


def standardize_and_limit_outliers_returns_first_try(dt_model, rol_freq, **kwargs):
    """
    Compute daily returns, standardize the returns and
    limit the returns if there are outliers in both
    positive and negative side.

    Args:
        dt_model (pd.DataFrame) : historical dataset
        rol_freq (int) : past return used
        **kwargs: training and validation test split
    """
    i = 0
    for index, row in dt_model.iterrows():
        data_series = row.loc[range(1, rol_freq + 1)].values
        # limit outliers by setting lower and upper range
        dt_median = np.median(data_series)
        dt_abs_spread_median = np.median(np.abs(data_series - dt_median))
        upper_range = dt_median + 5 * dt_abs_spread_median
        lower_range = dt_median - 5 - dt_abs_spread_median
        data_series = np.clip(data_series, lower_range, upper_range)
        # normalizing the data series, (not including target variable)
        mean = np.mean(data_series)
        st_dev = np.std(data_series)
        data_series = (data_series - mean) / st_dev
        dt_model.iloc[i, 0:rol_freq] = data_series
        i += 1
    df_training = dt_model[dt_model.index <= kwargs['training_end']]
    df_validation = dt_model[
        (dt_model.index > kwargs['training_end']) & (dt_model.index <= kwargs['validation_end'])]
    X_train = df_training.iloc[:, :rol_freq].values
    X_val = df_validation.iloc[:, :rol_freq].values
    Y_train = df_training['target'].values
    Y_val = df_validation['target'].values

    return X_train, Y_train, X_val, Y_val


def standardize_and_limit_outliers_returns(dt_model, rol_freq, **kwargs):
    """
    Compute daily returns, standardize the returns and
    limit the returns if there are outliers in both
    positive and negative side.

    Args:
        dt_model (pd.DataFrame) : historical dataset
        rol_freq (int) : past return used
        **kwargs: training and validation test split

    Returns:
        dataframes
    """
    # split the original dataset into 3 sets
    df_training = dt_model[dt_model.index <= kwargs['training_end']]
    df_validation = dt_model[
        (dt_model.index > kwargs['training_end']) & (dt_model.index <= kwargs['validation_end'])]
    X_train = df_training.iloc[:, :rol_freq].values
    X_val = df_validation.iloc[:, :rol_freq].values
    Y_train = df_training['target'].values
    Y_val = df_validation['target'].values
    # Normalize the feature vectors in the training set
    scaler = StandardScaler()
    X_train_norm = scaler.fit_transform(X_train)
    X_val_norm = scaler.transform(X_val)

    return X_train_norm, Y_train, X_val_norm, Y_val, scaler


def train_validation_test_split(df_hist, **kwargs):
    """
    Split data into training and validation test. Note that as
    the data is time series, so, split will not be random. Keep
    1 year of data for validation purpose and last 1 year of data
    for testing purpose and rest all for training.

    Args:
        df_hist (pd.DataFrame): historical complete data
        **kwargs: training and validation test split

    Returns:
        training and validation dataset
    """
    df_hist['daily_returns'] = df_hist['Adj Close'].pct_change()
    df_hist = df_hist.dropna()
    # create rolling dataset
    dt_model = create_features_and_target_split(pd.DataFrame(df_hist['daily_returns']),
                                                kwargs['past_day_returns_for_predicting'])
    # normalize and remove outliers

    return standardize_and_limit_outliers_returns(dt_model, kwargs['past_day_returns_for_predicting'], **kwargs)


def pre_process_for_test_set(daily_returns):
    """
    This method normalises returns
    and removes outliers. This approach is same
    as what we used while creating training dataset
    to make predictions on aligned data.

    Args:
        daily_returns (numpy array): daily return array

    Returns:
        numpy array
    """
    dt_median = np.median(daily_returns)
    dt_abs_spread_median = np.median(np.abs(daily_returns - dt_median))
    upper_range = dt_median + 5 * dt_abs_spread_median
    lower_range = dt_median - 5 - dt_abs_spread_median
    limit_returns = np.clip(daily_returns, lower_range, upper_range)
    mean = np.mean(limit_returns)
    st_dev = np.std(limit_returns)
    limit_returns = (limit_returns - mean) / st_dev

    return limit_returns
