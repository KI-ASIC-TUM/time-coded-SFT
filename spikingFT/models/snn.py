#!/usr/bin/env python3
"""
Module containing the abstract class defining the SNN classes API
"""
# Standard libraries
from abc import ABC, abstractmethod
# Local libraries
import spikingFT.utils.ft_utils


class FourierTransformSNN(ABC):
    """
    Abstract class defining the interface of the SNN implementations 

    Any SNN model that has to be run in the library shall be created as
    an instance of this class
    """
    def __init__(self, **kwargs):
        """
        Initialize network

        Parameters:
            nsamples (int): number of samples in radar chirp
            sim_time (int): Number of steps per network charging/spiking stage 
        """
        self.output = None
        self.nsamples = kwargs.get("nsamples")
        self.sim_time = kwargs.get("sim_time")
        self.nlayers = 1
        # WEIGHTS
        self.real_weights, self.imag_weights = self.calculate_weights()
        return

    def calculate_weights(self):
        re_weights, im_weights = spikingFT.utils.ft_utils.dft_connection_matrix(
            self.nsamples,
            self.PLATFORM
        )
        return re_weights, im_weights

    @abstractmethod
    def run(self, data, *args):
        return self.output

    def __call__(self, data, *args):
        self.run(*args)
        return
