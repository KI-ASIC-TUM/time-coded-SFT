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
import spikingFT.models.snn_radix4_loihi
import spikingFT.models.snn_radix4_brian
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
        self.metrics = {}

        self.raw_data = self.get_data()[:, :, :, 0]

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
        logger.info("No. time steps: {0}".format(sim_time))

        data = np.hstack([self.data.real, self.data.imag])

        encoder = spikingFT.utils.encoding.TimeEncoder(t_max=sim_time,
                                                       x_max=data.max(),
                                                       x_min=data.min()
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
        snn_config.pop("experiment", None)
        snn_config["nsamples"] = self.config["data"]["samples_per_chirp"]

        # Parse the simulation framework and instantiate the corresponding class
        framework = snn_config["framework"]
        mode = snn_config["mode"]
        if framework == "loihi" and mode == 'dft':
            snn = spikingFT.models.snn_loihi.SNNLoihi(**snn_config)
        elif framework == "brian" and mode == 'dft':
            snn = spikingFT.models.snn_brian.SNNBrian(**snn_config)
        elif framework == "numpy":
            snn = spikingFT.models.snn_numpy.SNNNumpy(**snn_config)
        elif framework == "loihi" and mode == 'fft':
            snn = spikingFT.models.snn_radix4_loihi.SNNRadix4Loihi(**snn_config)
        elif framework == "brian" and mode == 'fft':
            snn = spikingFT.models.snn_radix4_brian.SNNRadix4Brian(**snn_config)
        else:
            raise ValueError("Invalid framework: {}".format(framework))
        return snn

    def run_snn(self):
        """
        Execute the SNN simulation by feeding the provided data
        """
        logger.info("Running SNN simulation")
        output = []
        for frame in range(self.config["data"]["nframes"]):
            output.append(self.snn.run(self.encoded_data[frame]))
        # Return first frame
        return output[0]

    def test(self):
        """
        Measure accuracy metrics of the network

        numpy.fft library is used as reference for the error metrics
        """
        # Get reference FT result from NumPy on the same format as the output
        ftnp = np.fft.fft(self.data)
        ref = np.vstack((ftnp.real, ftnp.imag)).transpose()
        # Calculate the metrics
        rmse = spikingFT.utils.metrics.get_rmse(self.output, ref)
        rel_error = spikingFT.utils.metrics.get_error_hist(self.output, ref)
        self.metrics["rmse"] = rmse
        self.metrics["rel_error"] = rel_error
        logger.debug("Resulting RMSE: {}".format(self.metrics["rmse"]))
        return self.metrics

    def run(self, chirp_n=0):
        """
        Routine for initializing and running the SNN with the desired params
        """
        # Load encoded data
        self.data = self.raw_data[:, chirp_n, :]
        # Reduce data dimensionality, by ignoring chirp and antenna dimensions
        self.encoded_data = self.encode_data()
        # Instantiate the snn class with the specified configuration
        self.snn = self.initialize_snn()
        # Run the SNN with the collected data
        self.output = self.run_snn()
        self.test()
        logger.info("Execution finished")
        return self.output
