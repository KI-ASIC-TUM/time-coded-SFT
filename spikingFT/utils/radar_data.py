#!/usr/bin/env python3
"""
Main description of the module.
"""
# Standard libraries
import numpy as np
import matplotlib.pyplot as plt
import random

# Local and 3rd party libraries
import encoding


class RadarData:
    """Radar Data class for the Loihi SDFT Network
    Attributes:
        sim_time (int): time steps pf charging phase of SDFT -> time frame for the ttfs encoding
        raw: raw radar data
        nsamples (int): determined by number of samples in radar data;
        real_encoded_data (array): dimensions (nsamples/1) ttfs encoding
        imag_encoded_data (array): dimensions (nsamples/1) ttfs encoding
    """
    def __init__(self, file_name, sim_time, frame=None, chirp=None, channel=None):
        """ Prepares data for SDFT network
        Parameters:
            file_name (str): location of raw data file
            sim_time (int): time steps pf charging phase of SDFT -> time frame for the ttfs encoding
            frame (int): default random; specific frame can be chosen; range between 0-111
            chirp (int): default random; specific chirp can be chosen; range between 0-64
            channel (int): default random; specific channel can be chosen; range between 0-8
        """
        self.sim_time = sim_time
        data = np.load(file_name)['arr_0']
        # data: (frame, chirp, sample, channel)
        random_frame = random.randint(0, data.shape[0]-1) if frame is None else frame
        random_chirp = random.randint(0, data.shape[1]-1) if chirp is None else chirp
        random_channel = random.randint(0, data.shape[3]-1) if channel is None else channel
        self.raw = data[random_frame, random_chirp, :, random_channel]
        real_raw = self.raw.real
        imag_raw = self.raw.imag
        self.nsamples = self.raw.shape[0]
        self.real_encoded_data = self.encoded_data(real_raw)
        self.imag_encoded_data = self.encoded_data(imag_raw)

    def sinus_data_points(self):
        timesteps = np.linspace(0, self.nsamples, self.nsamples)
        real_raw = 1 + np.sin(0.2*np.pi*timesteps)
        imag_raw = real_raw
        self.real_encoded_data = self.encoded_data(real_raw)
        self.imag_encoded_data = self.encoded_data(imag_raw)

    def encoded_data(self, data):
        """ Encodes real and imag coefficients using the ttfs Encoder class
        Parameters:
            data (array): (nsamples,1) raw data
        Returns:
            encoded_data array
        """
        sim_time = self.nsamples
        time_encoder = encoding.TimeEncoder(t_max=sim_time, x_max=data.max())
        encoded_data = time_encoder.run(data)
        # Round values to unitary time steps
        return np.round(encoded_data, 0).astype(np.uint)

    def visualize(self):
        timesteps = np.linspace(0, self.nsamples, self.nsamples)
        plt.plot(timesteps, self.raw)
        plt.show()
        # Plot the positive part of the FFT spectrum (remove offset bin)
        ft = np.abs(np.fft.rfft(self.raw))
        plt.plot(ft[1:])
        plt.show()

