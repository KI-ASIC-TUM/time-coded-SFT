#!/usr/bin/env python3
"""
Module containing functions for loading data from local folder
"""
# Standard libraries
import logging
import numpy as np
import pathlib

logger = logging.getLogger('spiking-FT')


def get_source(dpath):
    """
    Retrieve which source generated the data in the specified path
    """
    if dpath.name == "TI_radar":
        source = "TI_sensor"
    elif dpath.name == "special_cases":
        source = "TI_sensor_special"
    elif dpath.name == "BBM":
        source = "BBM"
    else:
        raise ValueError("Invalid data path: {}".format(dpath))
    return source


def bbm_get_datacube():
    """
    Read file with the BBM simulator data and arrange it in a data cube
    """
    rootpath = pathlib.Path(__file__).parent.parent.parent
    filename = rootpath.joinpath("data/BBM/samples_ch_1.txt")
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
    # Reshape the data array so it incudes nframes and nantennas (1 each)
    nsamples, nchirps = data_cube.shape
    data_cube = data_cube.reshape([1, nchirps, nsamples, 1])
    return data_cube


def ti_get_datacube():
    """
    Read file with the TI dataset and arrange it in a data cube
    """
    rootpath = pathlib.Path(__file__).parent.parent.parent
    filename = rootpath.joinpath("data/TI_radar/1024/corner_reflector_1.npz")
    data_cube = np.load(filename)['arr_0']
    return data_cube


def ti_special_get_datacube():
    """
    Read file with TI sensor special cases and arrange it in a data cube
    """
    rootpath = pathlib.Path(__file__).parent.parent.parent
    filename = rootpath.joinpath("data/TI_radar/special_cases/data_tum.npy")
    raw_data = np.load(filename)
    # Add extra dimensions for having the standard data format
    data_cube = raw_data.reshape((1, raw_data.shape[0], raw_data.shape[1], 1))
    return data_cube


def get_data(source, config):
    """
    Load a datacube and collect a subset based on provided config
    """
    # Dataset dimensions limits
    if source == "TI_sensor":
        max_frames = 120
        max_antennas = 8
        max_chirps = 64
        max_samples = 1024
    elif source == "TI_sensor_special":
        max_frames = 1
        max_antennas = 1
        max_chirps = 4
        max_samples = 1024
    elif source == "BBM":
        max_frames = 1
        max_antennas = 1
        max_chirps = 128
        max_samples = 1024
    # Load configuration
    nframes = config["nframes"]
    nantennas = config["antennas"]
    nchirps = config["chirps_per_frame"]
    nsamples = config["samples_per_chirp"]

    # Check that provided config is compatible with the dataset
    if not 1 <= nframes <= max_frames:
        err_msg = "Invalid number of frames: {}. ".format(nframes)
        err_msg += "Maximum amount of frames is {}".format(max_frames)
        raise ValueError(err_msg)
    if not 1 <= nantennas <= max_antennas:
        err_msg = "Invalid number of antennas: {}. ".format(nantennas)
        err_msg += "Maximum amount of antennas is {}".format(max_antennas)
        raise ValuError(err_msg)
    if not 1 <= nchirps <= max_chirps:
        err_msg = "Invalid number of chirps: {}. ".format(nchirps)
        err_msg += "Maximum amount of chirps is {}".format(max_chirps)
        raise ValueError(err_msg)
    if not 1 <= nsamples <= max_samples:
        err_msg = "Invalid number of samples per chirp: {}. ".format(nsamples)
        err_msg += "Maximum amount of samples is {}".format(max_samples)
        raise ValueError(err_msg)

    # Load the data from corresponding sensor
    if source == "TI_sensor":
        datacube = ti_get_datacube()
    elif source == "TI_sensor_special":
        datacube = ti_special_get_datacube()
    elif source == "BBM":
        datacube = bbm_get_datacube()
    data = datacube[:nframes, :nchirps, :nsamples, :nantennas]
    msg = "Data loaded:\n- Source: {}\n- Nº frames: {}".format(source, nframes)
    msg += "\n- Nº chirps: {}\n- Nº samples: {}".format(nchirps, nsamples)
    msg += "\n- Nº antennas: {}".format(nantennas)
    logger.info(msg)
    return data


def load(dpath, config):
    """
    Load the data from the provided path
    """
    source = get_source(dpath)
    data = get_data(source, config)
    return data
