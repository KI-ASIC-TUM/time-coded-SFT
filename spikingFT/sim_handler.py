#!/usr/bin/env python3
"""
Module for initializing and launching the SNN simulation
"""
# Standard libraries
import logging
import numpy as np
# Local libraries
import spikingFT.models.snn_brian
import spikingFT.models.snn_loihi
import spikingFT.models.snn_numpy
import spikingFT.utils.load_data
import spikingFT.utils.encoding
import spikingFT.utils.metrics


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
        self.output = None
        self.snn = None
        self.performance = {}

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
        logger.info("Encoding data to spikes")
        sim_time = self.config["snn_config"]["sim_time"]

        encoder = spikingFT.utils.encoding.TimeEncoder(t_max=sim_time,
                                                       x_max=self.data.max(),
                                                       x_min = self.data.min()
                                                      )
        encoded_data = encoder.run(self.data)
        return encoded_data

    def initialize_snn(self):
        """
        Instantiation SNN simulation class and initialize its configuration
        """
        logger.info("Initializing SNN simulation")

        # Parse and edit the SNN configuration with required parameters
        snn_config = self.config["snn_config"]
        snn_config["nsamples"] = self.config["data"]["samples_per_chirp"]

        # Parse the simulation framework and instantiate the corresponding class
        framework = snn_config["framework"]
        if framework == "loihi":
            snn = spikingFT.models.snn_loihi.SNNLoihi(**snn_config)
        elif framework == "brian":
            snn = spikingFT.models.snn_brian.SNNBrian(**snn_config)
        elif framework == "numpy":
            snn = spikingFT.models.snn_numpy.SNNNumpy(**snn_config)
        else:
            raise ValueError("Invalid framework: {}".format(framework))
        return snn

    def parse_results(self, result):
        """
        Parse SNN output, generate plots, and save results
        """
        logger.info("Parsing results of SNN simulation")
        return result

    def run_snn(self):
        """
        Execute the SNN simulation by feeding the provided data
        """
        logger.info("Running SNN simulation")
        output = []
        result = []
        for frame in range(self.config["data"]["nframes"]):
            output.append(self.snn.run(self.encoded_data[frame]))
            result.append(self.parse_results(output))
        # Return first frame
        return result[0]

    def test(self):
        """
        Measure the accuracy of the network

        numpy.fft library is used as reference for the error metrics
        """
        # Get reference FT result from NumPy
        ft_np = np.fft.fft(self.data[0, 0, :, 0])
        ft_np_modulus = np.abs(ft_np)[1:int(self.snn.nsamples/2)]
        rmse = spikingFT.utils.metrics.get_rmse(self.output, ft_np_modulus)
        self.performance["rmse"] = rmse
        import pdb; pdb.set_trace()
        return self.performance

    def run(self):
        """
        Routine for initializing and running the SNN with the desired params
        """
        # Load encoded data
        self.data = self.get_data()
        # Reduce data dimensionality, by ignoring chirp and antenna dimensions
        self.encoded_data = self.encode_data()[:, 0, :, 0]
        # Instantiate the snn class with the specified configuration
        self.snn = self.initialize_snn()
        # Run the SNN with the collected data
        self.output = self.run_snn()
        self.test()
        return self.output
