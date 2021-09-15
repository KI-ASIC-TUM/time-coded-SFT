#!/usr/bin/env python3
"""
Module one-line definition
"""
# Standard libraries
import matplotlib.pyplot as plt
import numpy as np
# Local libraries
import spikingFT.startup


if __name__ == "__main__":
    conf_file = "../config/test_experiment.json"
    sim = spikingFT.startup.startup(conf_file)

    # Plot standard 1D FFT
    chirp_data = sim.encoded_data[0, :]
    fft = np.abs(np.fft.fft(chirp_data))
    plt.plot(fft[1:int(fft.size/2)])
    plt.show()

    #TODO: Plot output from S-FT
