"""
This module runs other modules which creates a dataset
that can be used to compute implied volatility using the brent
method for a given black-scholes price and then
build a neural network model learn non-linear relationship.

This model is further used combine with Heston NN model
to understand how well we can replicate numerical methods
and to construct volatility smile-skew observed in the real market.

Note that the model is built on learning the relationship between
time value of option (subtracting intrinsic value) as per section 4.3
of the literature document
"""
import json
import argparse

from model import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Take number of data required for running model', add_help=False)
    parser.add_argument('-t', '--num_dt_training', type=int, help='Actual model run dataset size')
    args = parser.parse_args()
    df, st_current_price, range_of_inputs = pre_processing(args.num_dt_training, "BS")
    big_dataset = create_implied_vol_dataset(df, st_current_price, range_of_inputs)
    file_name = r"./model/output/" + "best_hyper_parameter.json"
    with open(file_name) as f:
        df_hyper = json.load(f)
    feature_columns = ['moneyness', 'time_to_maturity', 'risk_free_rate', 'scaled_time_value']
    target = 'volatility'
    model = run_nn_model(big_dataset, df_hyper, feature_columns, target)
    model_save_path = r"./model/output/" + "implied_vol_NN_model.h5"
    model.save(model_save_path)
