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
import spikingFT.utils.encoding

logger = logging.getLogger('spiking-FT')


class SimHandler():
    """
    Class for handling the simulation workflow
    """
    def __init__(self, datapath, config):
        """
        Initialization
        """
        self.datapath = datapath
        self.config = config
        self.data = None
        self.encoded_data = None
        self.snn = None

    def get_data(self):
        """
        Load the simulation data from the specified path
        """
        logger.info("Loading data")

        sensor_config = self.config["data"]
        data = spikingFT.utils.load_data.load(self.datapath, sensor_config)
        return data

    def encode_data(self):
        """
        Encode sensor data using TTFS method. Range is defined by config
        """
        sim_time = self.config["snn_config"]["iterations"]
        encoder = spikingFT.utils.encoding.TimeEncoder(t_max=sim_time,
                                                       x_max=self.data.max()
                                                      )
        encoded_data = encoder.run(self.data)
        return encoded_data

    def initialize_snn(self):
        """
        Instantiation SNN simulation class and initialize its configuration
        """
        logger.info("Initializaing SNN simulation")

        framework = self.config["snn_config"]["framework"]
        if framework == "loihi":
            snn = spikingFT.models.snn_loihi.SNNLoihi()
        elif framework == "brian":
            snn = spikingFT.models.snn_brian.SNNBrian()
        else:
            raise ValueError(
                    "Invalid simulation framework: {}".format(framework)
                    )
        return snn

    def parse_results(self, result):
        """
        Parse SNN output, generate plots, and save results
        """
        return

    def run_snn(self):
        """
        Execute the SNN simulation by feeding the provided data
        """
        logger.info("Running SNN simulation")

        # result = snn.run(data)
        result = None
        self.parse_results(result)
        return

    def run(self):
        """
        Routine for initializing and running the SNN with the desired params
        """
        # Load encoded data
        self.data = self.get_data()
        self.encoded_data = self.encode_data()
        # Instantiate the snn class with the specified configuration
        self.snn = self.initialize_snn()
        # Run the SNN with the collected data
        self.run_snn()
        return
