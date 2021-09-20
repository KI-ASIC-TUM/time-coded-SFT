#!/usr/bin/env python3
"""
Script for testing the SNNNumpy class with sample data
"""
# Standard libraries
import matplotlib.pyplot as plt
import numpy as np
# Local libraries
import spikingFT.startup


def main(conf_filename="../config/test_experiment.json"):
    # Instantiate a simulation handler and run spiking FT with sample data
    sim_handler = spikingFT.startup.startup(conf_filename)
    nsamples = sim_handler.snn.nsamples
    real_spikes = sim_handler.output[0][:, 0]
    imag_spikes = sim_handler.output[0][:, 1]
    sft_modulus = np.sqrt(real_spikes**2 + imag_spikes**2)[1:int(nsamples/2)]

    # Get reference FT result from NumPy
    ft_np = np.fft.fft(sim_handler.data[0, 0, :, 0])
    ft_np_modulus = np.abs(ft_np)[1:int(nsamples/2)]

    # Plot SFT result and reference result
    plt.subplot(3, 1, 1)
    for sample in range(1,5):
        plt.plot(sim_handler.snn.voltage[:, sample, 0])
        plt.plot(sim_handler.snn.voltage[:, sample, 1])
    plt.title("Membrane voltages over simulation time")
    plt.subplot(3, 1, 2)
    plt.plot(sft_modulus)
    plt.title("NumPy FFT")
    plt.subplot(3, 1, 3)
    plt.plot(ft_np_modulus)
    plt.title("NumPy-based Spiking DFT")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
