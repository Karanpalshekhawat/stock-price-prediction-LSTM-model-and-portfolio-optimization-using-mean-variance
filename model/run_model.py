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


