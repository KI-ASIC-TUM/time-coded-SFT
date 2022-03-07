#!/usr/bin/env python3
"""
Module containing the abstract class defining the SNN classes API
"""
# Standard libraries
from abc import ABC, abstractmethod
# Local libraries
import spikingFT.utils.ft_utils
import numpy as np


class FourierTransformSNN(ABC):
    """
    Abstract class defining the interface of the SNN implementations 

    Any SNN model that has to be run in the library shall be created as
    an instance of this class.
    The constant FFT defines whether the DFT or FFT version is used.
    """
    FFT = None
    def __init__(self, **kwargs):
        """
        Initialize network

        Parameters:
            nsamples (int): number of samples in radar chirp
            sim_time (int): Number of steps per network charging/spiking stage
            l_weights (array): list of weight matrices or real[0] and imaginary[1] weight matrix
            output (array): output spikes of the network
        """
        self.output = None
        self.nsamples = kwargs.get("nsamples")
        self.nlayers = int(np.log(self.nsamples)/np.log(4)) if self.FFT else 1

        #TODO: take care of sim_time and cycle time
        self.sim_time = kwargs.get("sim_time")

        # WEIGHTS
        if self.FFT:
            # Sparse weight matrices (radix4)
            # One matrix includes real and imaginary parts (stacked)
            self.l_weights = self.calculate_weights_fft()
        else:
            # Real[0] and imaginary[1] components of dft weight matrix
            self.l_weights = self.calculate_weights_dft()

        return

    def calculate_weights_fft(self):

        weight_matrices = []

        for l in range(self.nlayers):
            
            weight_matrix = spikingFT.utils.ft_utils.fft_connection_matrix(
                layer=l,
                nsamples=self.nsamples,
                platform = self.PLATFORM
            )
            weight_matrix = spikingFT.utils.ft_utils.normalize(weight_matrix,
                    self.PLATFORM)
            weight_matrices.append(weight_matrix)

        return weight_matrices

    def calculate_weights_dft(self):
        re_weights, im_weights = spikingFT.utils.ft_utils.dft_connection_matrix(
            self.nsamples,
            self.PLATFORM
        )
        
        return [re_weights, im_weights]

    @abstractmethod
    def run(self, data, *args):
        return self.output

    def __call__(self, data, *args):
        self.run(*args)
        return
