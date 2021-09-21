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
    real_spikes = sim_handler.output[:, 0][1:int(nsamples/2)]
    imag_spikes = sim_handler.output[:, 1][1:int(nsamples/2)]
    sft_modulus = np.sqrt(real_spikes**2 + imag_spikes**2)

    # Get reference FT result from NumPy
    ft_np = np.fft.fft(sim_handler.data[0, 0, :, 0])
    ft_np_modulus = np.abs(ft_np)[1:int(nsamples/2)]

    # Plot S-FT result and reference result
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

    # Plot relative error histogram
    rel_error = sim_handler.metrics["rel_error"]
    real_error = rel_error[:, 0]
    imag_error = rel_error[:, 1]
    # Real spectrum
    ax_left = plt.subplot(3, 1, 1)
    ax_right = ax_left.twinx()
    l1 = ax_left.plot(real_spikes, color="blue", label="signal")
    l2 = ax_right.plot(real_error, color="red", linestyle="--", label="error")
    ax_right.set_ylim([0., 0.05])
    ax_left.set_ylabel("|FT(x)|")
    ax_right.set_ylabel("Rel. error")
    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax_left.legend(lines, labels)
    ax_left.set_title("Real spectrum rel. error")
    #Left spectrum
    ax_left = plt.subplot(3, 1, 2)
    ax_right = ax_left.twinx()
    l1 = ax_left.plot(imag_spikes, color="blue", label="signal")
    l2 = ax_right.plot(imag_error, color="red", linestyle="--", label="error")
    ax_right.set_ylim([0., 0.05])
    ax_left.set_ylabel("|FT(x)|")
    ax_right.set_ylabel("Rel. error")
    ax_left.set_title("Imaginary spectrum rel. error")
    # Modulus
    abs_error = (real_error + imag_error) / 2
    ft_modulus = np.sqrt(real_spikes**2 + imag_spikes**2)
    ax_left = plt.subplot(3, 1, 3)
    ax_right = ax_left.twinx()
    l1 = ax_left.plot(ft_modulus, color="blue", label="signal")
    l2 = ax_right.plot(abs_error, color="red", linestyle="--", label="error")
    ax_right.set_ylim([0., 0.05])
    ax_left.set_ylabel("|FT(x)|")
    ax_right.set_ylabel("Rel. error")
    ax_left.set_title("Imaginary spectrum rel. error")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
