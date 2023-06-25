"""
This module is added to define methods
used in identifying weights for the stocks
included in analysis using the mean variance model.
"""

import numpy as np
from scipy.optimize import minimize


def minimize_variance_for_given_return(num_of_stocks, exp_return, covar_matrix):
    """

    Args:
        num_of_stocks (int): number of stocks used in the analysis.
        exp_return (np.array): expected return of each stock
        covar_matrix (np.array): covariance matrix computed using the expected returns

    Returns:
         np.array
    """
    weights = np.ones(num_of_stocks) / num_of_stocks  # Initialize equal weights

    def portfolio_variance():
        """
        This method computes portfolio variance for a given weights
        of each stock in portfolio and covariance matrix.
        """
        return np.dot(np.dot(weights, covar_matrix), weights.T)

    def weight_sum_constraint():
        """
        This method adds the first constraint of summation
        of weights to be equal to 1.
        """
        return np.sum(weights) - 1

    def expected_portfolio_return_constraint():
        """
        This method adds the second constraint of expected
        return of a portfolio.
        """
        return np.dot(weights, exp_return) - 1

    # Optimization using scipy
    cons = ({'type': 'eq', 'fun': weight_sum_constraint}, {'type': 'eq', 'fun': expected_portfolio_return_constraint})
    bounds = [(0, 1) for _ in range(num_of_stocks)]  # Bounds: each weight between 0 and 1
    result = minimize(portfolio_variance, weights, method='SLSQP', bounds=bounds, constraints=cons)

    optimal_weights = result.x

    return optimal_weights
