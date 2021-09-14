#!/usr/bin/env python3
"""
Module for initializing and launching the SNN simulation
"""
# Standard libraries
import logging
# Local libraries
import spikingFT.models.snn_brian
import spikingFT.models.snn_loihi
import spikingFT.utils.load_data

logger = logging.getLogger('spiking-FT')


def get_data(datapath, config):
    """
    Load the simulation data from the specified path
    """
    logger.info("Loading data")

    sensor_config = config["data"]
    data = spikingFT.utils.load_data.load(datapath, sensor_config)
    return data


def initialize_snn(config):
    """
    Instantiation SNN simulation class and initialize its configuration
    """
    logger.info("Initializaing SNN simulation")

    framework = config["snn_config"]["framework"]
    if framework == "loihi":
        snn = spikingFT.models.snn_loihi.SNNLoihi()
    elif framework == "brian":
        snn = spikingFT.models.snn_brian.SNNBrian()
    else:
        raise ValueError("Invalid simulation framework: {}".format(framework))
    return snn


def parse_results(result):
    """
    Parse SNN output, generate plots, and save results
    """
    return


def run_snn(snn, data):
    """
    Execute the SNN simulation by feeding the provided data
    """
    logger.info("Running SNN simulation")

    # result = snn.run(data)
    result = None
    parse_results(result)
    return


def run(datapath, config):
    """
    Routine for initializing and running the SNN with the desired params
    """
    # Load encoded data
    data = get_data(datapath, config)
    # Instantiate the snn class with the specified configuration
    snn = initialize_snn(config)
    # Run the SNN with the collected data
    run_snn(snn, data)
    return
