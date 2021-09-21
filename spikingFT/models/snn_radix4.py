#!/usr/bin/env python3
"""
Module containing the abstract class defining the SNN classes API
"""
# Standard libraries
from abc import ABC, abstractmethod
import numpy as np
# Local libraries
import spikingFT.utils.ft_utils


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
        # WEIGHTS and BIASES
        self.l_weights, self.l_biases = self.calculate_weights()
        return

    def calculate_weights(self):

        weight_matrices = []
        biases = []

        for l in range(self.nlayers):

            weight_matrix = spikingFT.utils.ft_utils.fft_connection_matrix(
                layer=l,
                nsamples=self.nsamples,
            )
            weight_matrices.append(weight_matrix)
            biases.append(np.sum(weight_matrix, axis=1)*self.sim_time/2)

        return weight_matrices, biases

    @abstractmethod
    def run(self, data, *args):
        return self.output

    def __call__(self, data, *args):
        self.run(*args)
        return
