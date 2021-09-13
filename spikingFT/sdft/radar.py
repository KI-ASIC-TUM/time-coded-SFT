import numpy as np
import json
import pathlib

import encoding


def get_datacube(filename):
    """
    Read file with the BBM simulator data and arrange it in a data cube
    """
    data_cube = None
    with open(str(filename), "r") as f:
        data = f.readlines()
    for row in data:
        row = row.split(",")
        row[-1] = row[-1].strip("\n")
        row_arr = np.array(row).astype(np.float)
        # Construct the data cube. First iteration creates an array
        # following iterations stack on top
        if data_cube is None:
            data_cube = np.array(row_arr)
        else:
            data_cube = np.vstack((data_cube, row_arr))
    # Only the 900 first samples contain information
    data_cube = data_cube.transpose()[:, :900]
    return data_cube


def bbm_read(path):
    """
    Extract output of BBM. Either single of multiple channel files
    """
    filenames = []
    if path.is_file():
        filenames = [path]
    else:
        filenames = [file for file in path.iterdir() if file.name[-4:]==".txt"]
    data_cubes = [get_datacube(fname) for fname in filenames]
    return data_cubes


def load_config(config_file):
    """
    Load the configuration file with the simulation parameters

    @param conf_file: str with the relative address of the config file
    @param dims: Number of Fourier dimensions of the experiment
    """
    # Load configuaration data from local file
    with open(config_file) as f:
        config_data = json.load(f)
    fname = config_data["filename"]
    fpath = pathlib.Path().resolve().joinpath(fname)
    print(fpath)
    dft_args = config_data["dft_encoding_parameters"]
    return fpath, dft_args


def normalize(input_data):
    """
    Convert the input data to the range [0..1]
    """
    min_value = input_data.min()
    # Remove bias from the input signal
    no_offset = input_data - min_value
    # Normalize to [0..1]
    max_value = no_offset.max()
    normalized = no_offset / max_value
    return normalized


def linear_rate_encoding(raw_data, encoding_params):
    """
    Normalize and encode input data using the LinearFrequencyEncoder
    """
    # Normalize all samples between 0 and 1, based on global max and min values
    normalized_cube = normalize(raw_data)
    # Encode the voltage to spikes using rate encoding
    encoder = encoding.LinearFrequencyEncoder(**encoding_params, random_init=True)
    encoded_cube = encoder(normalized_cube)
    return encoded_cube


def data_encoding(raw_data, coding_params):
    """
    Normalize and encode input data using the LinearFrequencyEncoder
    """
    # Normalize all samples between 0 and 1, based on global max and min values
    normalized_cube = normalize(raw_data)
    encoder = encoding.TimeEncoder(**coding_params)
    encoded_data = encoder(normalized_cube)
    return encoded_data