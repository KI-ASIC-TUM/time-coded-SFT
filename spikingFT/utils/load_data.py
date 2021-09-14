#!/usr/bin/env python3
"""
Module one-line definition
"""
# Standard libraries
import numpy as np
import pathlib
# Local libraries


def get_source(dpath):
    """
    Retrieve which source generated the data in the specified path
    """
    if dpath.name == "TI_radar":
        source = "TI_sensor"
    elif dpath.name == "BBM":
        source = "BBM"
    else:
        raise ValueError("Invalid data path: {}".format(dpath))
    return source


def bbm_get_datacube():
    """
    Read file with the BBM simulator data and arrange it in a data cube
    """
    path = pathlib.Path(__file__).parent.parent.parent
    filename = path.joinpath("data/BBM/samples_ch_1.txt")
    data_cube = None
    with open(filename, "r") as f:
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
    data_cube = data_cube.transpose()
    return data_cube


def get_BBM_data(config):
    """
    Load BBM datacube and collect subset based on provided config
    """
    nframes = config["nframes"]
    nchirps = config["chirps_per_frame"]
    nsamples = config["samples_per_chirp"]
    # Check that provided config is compatible with BBM dataset
    if nframes != 1:
        err_msg = "Invalid number of frames: {}. ".format(nframes)
        err_msg += "BBM dataset only contains 1 frame"
        raise ValueError(err_msg)
    if not 1 <= nchirps <= 128:
        err_msg = "Invalid number of chirps: {}. ".format(nchirps)
        err_msg += "Amount of chirps must be between 1 and 128"
        raise ValueError(err_msg)
    if not 1 <= nsamples <= 1024:
        err_msg = "Invalid number of samples per chirp: {}. ".format(nsamples)
        err_msg += "Samples per chirp must be between 1 and 1024"
        raise ValueError(err_msg)
    # Load BBM datacube and collect the required amount of data
    datacube = bbm_get_datacube()
    data = datacube[:nchirps, :nsamples]
    return data


def get_TI_data(config):
    """
    Load a TI datacube and collect a subset based on provided config
    """
    raise NotImplementedError
    data = None
    return data


def get_data(dpath, source, config):
    if source == "TI_sensor":
        data = get_TI_data(config)
    elif source == "BBM":
        data = get_BBM_data(config)
    return data


def load(dpath, config):
    """
    Load the data from the provided path
    """
    source = get_source(dpath)
    data = get_data(dpath, source, config)

    return data
