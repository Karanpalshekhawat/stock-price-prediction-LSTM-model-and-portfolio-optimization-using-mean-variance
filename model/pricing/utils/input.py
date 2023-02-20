"""
In this module, I have added methods that
are used to generate range of input parameters
and then compute sampling using LHS
Also, it contains methods to read input variables
"""

import json
import yaml
import pandas as pd
from scipy.stats import qmc


def read_input_parameters_range(fltr):
    """
    Reading input parameters range to create set of
    dataset that will be used for training and testing
    purpose

    Args:
        fltr (str) : running BS or heston pricing model

    Returns:
        dict
    """
    pt = r"./model/static_data/"
    if fltr == "BS":
        json_file = pt + "parameters-bs-range.json"
    else:
        json_file = pt + "parameters-heston-range.json"

    with open(json_file) as f:
        data = json.load(f)

    return data


def create_set_of_input_parameters(nmb, fltr):
    """
    Using range of Input parameters defined in json file,
    a set of options parameters range is created using
    Latin hypercube sampling.

    Args:
        nmb (int) : number of option parameters to generate
        fltr (str) : running BS or heston pricing model

    Returns:
        pd.DataFrame
    """
    rng = read_input_parameters_range(fltr)
    dimension = len(rng)
    sampler = qmc.LatinHypercube(d=dimension)
    sample = sampler.random(n=nmb)
    l_bounds = [items[0] for items in rng.values()]
    u_bounds = [items[1] for items in rng.values()]
    sample_scaled = qmc.scale(sample, l_bounds, u_bounds)
    df = pd.DataFrame(sample_scaled, columns=list(rng.keys()))
    return df


def read_input_variables_file():
    """
    This method reads input variables yaml file
    and returns it as a dataframe
    """
    pt = r"./model/static_data/"
    yml_file = pt + "input_variables.yml"
    with open(yml_file) as file:
        df = pd.json_normalize(yaml.safe_load(file))

    return df


def read_hyper_parameters_range():
    """
    Reading input parameters range to create set of
    dataset that will be used for training and testing
    purpose

    Returns:
        dict
    """
    pt = r"./model/static_data/"
    json_file = pt + "hyper-parameters-range.json"
    with open(json_file) as f:
        data = json.load(f)

    return data
