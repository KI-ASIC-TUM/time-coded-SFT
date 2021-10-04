#!/usr/bin/env python3
"""
Module containing the abstract class defining the SNN classes API
"""
# Standard libraries
from abc import ABC, abstractmethod
import numpy as np
# Local libraries
import spikingFT.utils.ft_utils
import logging
logger = logging.getLogger('spiking-FT')


class FastFourierTransformSNN(ABC):
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
        #TODO: check consistency
        self.nlayers = int(np.log(self.nsamples)/np.log(4))

        #TODO: take care of sim_time and cycle time
        self.sim_time = kwargs.get("sim_time")
        # WEIGHTS
        self.l_weights = self.calculate_weights()
        return

    def calculate_weights(self):

        weight_matrices = []

        for l in range(self.nlayers):

            if self.PLATFORM == 'loihi':
                axis = 1
            elif self.PLATFORM =='brian':
                axis = 0
            else:
                axis = 0

            weight_matrix = spikingFT.utils.ft_utils.fft_connection_matrix(
                layer=l,
                nsamples=self.nsamples,
                platform = self.PLATFORM
            )
            weight_matrix = spikingFT.utils.ft_utils.normalize(weight_matrix,
                    self.PLATFORM)
            weight_matrices.append(weight_matrix)

        return weight_matrices

    @abstractmethod
    def run(self, data, *args):
        return self.output

    def __call__(self, data, *args):
        self.run(*args)
        return
