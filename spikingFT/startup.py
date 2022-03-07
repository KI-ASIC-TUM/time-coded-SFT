#!/usr/bin/env python3
"""
Module for loading metadata and starting the simulation
"""
# Standard libraries
import json
import logging
import pathlib
import time
# Local libraries
import spikingFT.sim_handler
import spikingFT.utils.parse_args


def load_config(conf_file):
    """
    Load the configuration file with the simulation parameters

    @param conf_file: str with the relative address of the config file
    @param dims: Number of Fourier dimensions of the experiment 
    """
    # Load configuaration data from local file
    with open(conf_file) as f:
        config_data = json.load(f)
    path = config_data["datapath"]
    datapath = pathlib.Path(__file__).resolve().parent.joinpath(path)
    # Load the configuration parameters
    config = config_data["config"]
    # Add experiment configuration, if exists
    try:
        config["experiment"] = config_data["experiment"]
    except KeyError:
        pass
    return datapath, config


def conf_logger():
    # Create log folder
    logpath = pathlib.Path(__file__).resolve().parent.parent.joinpath("log")
    pathlib.Path(logpath).mkdir(parents=True, exist_ok=True)
    datetime = time.strftime("%Y-%m-%d %H:%M:%S")
    fdatetime = time.strftime("%Y%m%d-%H%M%S")
    # Create logger
    logger = logging.getLogger('spiking-FT')
    logger.setLevel(logging.DEBUG)

    # Create file handler
    file_handler = logging.FileHandler("{}/{}.log".format(logpath, fdatetime))
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s',
                                  "%H:%M:%S")
    file_handler.setFormatter(formatter)
    file_handler.stream.write("{} MAIN PROGRAM EXECUTION\n".format(datetime))
    logger.addHandler(file_handler)

    # Create console handler
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(levelname)s: %(message)s')
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def run(datapath, config, autorun):
    """
    Run the algorithm with the loaded configuration
    """
    sim_handler = spikingFT.sim_handler.SimHandler(datapath, config)
    if autorun:
        sim_handler.run()
    return sim_handler


def startup(conf_file, autorun=True):
    """
    Run the DFT and CFAR on BBM data
    """
    datapath, config = load_config(conf_file)

    logger = conf_logger()
    msg = "Starting up spiking-FT:"
    msg += "\n- Configuration file: {}".format(conf_file)
    msg += "\n- Simulation time: {}".format(config["snn_config"]["sim_time"])
    msg += "\n- Time step: {}".format(config["snn_config"]["time_step"])
    msg += "\n- FT mode: {}".format(config["snn_config"]["mode"])
    msg += "\n- Framework: {}".format(config["snn_config"]["framework"])
    msg += "\n- Test performance: {}".format(config["snn_config"]["measure_performance"])
    logger.info(msg)

    sim_handler = run(datapath, config, autorun)
    return sim_handler


if __name__ == "__main__":
    conf_file, _, _ = spikingFT.utils.parse_args.parse_args()
    startup(conf_file)
