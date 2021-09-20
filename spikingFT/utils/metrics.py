#!/usr/bin/env python3
"""
Module containing accuracy metrics
"""
# Standard libraries
import numpy as np
# Local libraries


def get_mse(data, ref):
    """
    Calculate the mean square error of the data

    Both @data and @ref must be numpy arrays of the same shape
    """
    mse = ((data- ref)**2) / data.size
    return mse

def get_rmse(data, ref):
    """
    Calculate the root mean square error of the data

    Both @data and @ref must be numpy arrays of the same shape
    """
    mse = get_mse(data, ref)
    rmse = np.sqrt(mse)
    return rmse
