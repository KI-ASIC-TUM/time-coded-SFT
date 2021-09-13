#!/usr/bin/env python3
"""
Module for initializing and launching the SNN simulation
"""
# Standard libraries
# Local libraries
import spikingFT.models.snn_brian
import spikingFT.models.snn_loihi
import spikingFT.utils.radar_data

logger = logging.getLogger('spiking-FT')


def load_data(data_path):
    data = None
    return data


def initialize_snn(config):
    snn = None
    return snn


def parse_results(result)
    return


def run(data_path, config):
    # Load encoded data
    data = load_data(data_path)
    # Instantiate the snn class with the specified configuration
    snn = initialize_snn(config)
    # Run the SNN with the collected data
    result = snn.run(data)
    parse_results(result)
    return
