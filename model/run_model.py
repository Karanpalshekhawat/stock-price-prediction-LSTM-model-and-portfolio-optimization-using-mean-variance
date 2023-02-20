"""
This module is the main file which call all
other modules to import data, run neural
network model and generate parameters and then
create output file for pricing options
"""
import json
import argparse

from model import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Take number of data required for tuning and running model',
                                     add_help=False)
    parser.add_argument('-h', '--num_dt_hyper', type=int, help='Hyper parameter tuning dataset size')
    parser.add_argument('-t', '--num_dt_training', type=int, help='Actual model run dataset size')
    args = parser.parse_args()
    df, st_current_price, range_of_inputs = pre_processing(args.num_dt_hyper,
                                                           "BS")  # small dataset for hyperparameter tuning
    small_dt_set = create_dataset(df, st_current_price, range_of_inputs)
    running_hyperparameter_tuning = False
    if running_hyperparameter_tuning:
        hyperparameter_tuning(small_dt_set)
    file_name = r"./model/output/" + "best_hyper_parameter.json"
    with open(file_name) as f:
        df_hyper = json.load(f)
    df, st_current_price, range_of_inputs = pre_processing(args.num_dt_training, "BS")
    big_dataset = create_dataset(df, st_current_price, range_of_inputs)  # big dataset for NN model
    feature_columns = ['moneyness', 'time_to_maturity', 'risk_free_rate', 'volatility']
    target = 'opt_price_by_strike'
    model, training_history = run_nn_model(big_dataset, df_hyper, feature_columns, target)
    model_save_path = r"./model/output/" + "BS_NN_model.h5"
    model.save(model_save_path)
