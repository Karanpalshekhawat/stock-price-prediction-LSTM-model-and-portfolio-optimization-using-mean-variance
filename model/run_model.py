"""
This module is the main file which call all
other modules to run elastic net, SVR, Random Forest
and LSTM model.
"""

from model import *

if __name__ == "__main__":
    # run Elastic Net model for all stocks
    model, scaler = run_elastic_net_model_for_all_stocks()
