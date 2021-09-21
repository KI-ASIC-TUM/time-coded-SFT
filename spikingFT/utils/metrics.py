#!/usr/bin/env python3
"""
Module containing accuracy metrics
"""
# Standard libraries
import numpy as np
# Local libraries


def simplify_ft(data):
    """
    Remove irrelevant bins and normalize the FT data between 0 and 1

    The irrelevant bins are the offset bin and the bins that belong to
    the negative frequency spectrum.

    @data is a 2D (Nx2) np.array containing the output of a 1D, where N
    is the amount of samples per chirp, and each column represents the
    real and imaginary components, respectively.
    """
    # Remove offset and negative spectrum bins
    half_length = int(data.shape[0]/2)
    cropped = data[1:half_length, :]
    # Normalize resulting FT
    result = cropped - cropped.min()
    result /= result.max()
    return result


def get_mse(signal, ref):
    """
    Calculate the mean square error of the signal

    Both @signal and @ref must be numpy arrays of the same shape
    """
    signal = simplify_ft(signal)
    ref = simplify_ft(ref)
    quadratic_diff = ((signal- ref)**2)
    mse = quadratic_diff.sum() / ref.size
    return mse


def get_rmse(signal, ref):
    """
    Calculate the root mean square error of the signal

    Both @signal and @ref must be numpy arrays of the same shape
    """
    mse = get_mse(signal, ref)
    rmse = np.sqrt(mse)
    return rmse

def get_error_hist(signal, ref):
    """
    Get the histogram of the relative error along the output
    """
    # Add small value for avoding divide-by-zero situations
    signal = np.abs(simplify_ft(signal)) + 0.001
    ref = np.abs(simplify_ft(ref)) + 0.001
    # Get the absolute error of each bin.
    diff = np.abs(signal - ref)
    # Get the relative error by dividing by the ideal intensity
    relative_error = diff / ref
    return relative_error
