"""
This module is the main file which call all
other modules to run elastic net, Random Forest
and LSTM model.
"""

from model import *

from datetime import date, datetime

if __name__ == "__main__":
    start_date = date(2012, 1, 1)
    end_date = datetime.today()
    data_dict = get_historical_stock_data(start_date, end_date)
    # run Elastic Net model for all stocks
    elastic_net_model_details = run_elastic_net_model_for_all_stocks(data_dict, end_date)
    # run LSTM model for all stocks
    lstm_model_details = run_lstm_model_for_all_stocks(data_dict, end_date)
