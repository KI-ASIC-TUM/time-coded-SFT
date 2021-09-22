#!/usr/bin/env python3
"""
Script for testing the SNNNumpy class with sample data
"""
# Standard libraries
import matplotlib.pyplot as plt
import numpy as np
# Local libraries
import spikingFT.startup
import spikingFT.utils. plotter


def single_run_experiment(conf_filename="../config/test_experiment.json"):
    # Instantiate a simulation handler and run spiking FT with sample data
    sim_handler = spikingFT.startup.startup(conf_filename)
    nsamples = sim_handler.snn.nsamples
    real_spikes = sim_handler.output[:, 0][1:int(nsamples/2)]
    imag_spikes = sim_handler.output[:, 1][1:int(nsamples/2)]
    sft_modulus = np.sqrt(real_spikes**2 + imag_spikes**2)

    # Plot S-FT result and reference result
    kwargs = {}
    kwargs["plot_names"] = ["voltages", "spikes", "FT"]
    kwargs["data"] = [
        sim_handler.snn.voltage,
        (real_spikes, imag_spikes),
        sft_modulus
    ]
    sim_plotter = spikingFT.utils.plotter.SNNSimulationPlotter(**kwargs)
    sim_plotter()

    # Plot relative error histograms
    rel_error = sim_handler.metrics["rel_error"]
    real_error = rel_error[:, 0]
    imag_error = rel_error[:, 1]
    abs_error = (real_error + imag_error) / 2
    ft_modulus = np.sqrt(real_spikes**2 + imag_spikes**2)
    kwargs = {}
    kwargs["plot_names"] = ["real_spectrum", "imag_spectrum", "modulus"]
    kwargs["data"] = [
        (real_spikes, real_error),
        (imag_spikes, imag_error),
        (ft_modulus, abs_error)
    ]
    error_plotter = spikingFT.utils.plotter.RelErrorPlotter(**kwargs)
    error_plotter()
    return

def sim_times_experiment(conf_filename="../config/test_experiment_simtimes.json"):
    # Instantiate a simulation handler with specified configuration
    sim_handler = spikingFT.startup.startup(conf_filename, autorun=False)
    # Get simulation times from config file
    exp_config = sim_handler.config["experiment"]
    sim_times = exp_config["sim_times"]
    errors = []
    # Iterate over the different sim times and simulate network
    for sim_time in sim_times:
        sim_handler.config["snn_config"]["sim_time"] = sim_time
        sim_handler.run()
        errors.append(sim_handler.metrics["rmse"])
    # Plot results
    kwargs = {}
    kwargs["plot_names"] = ["single_nsamples"]
    kwargs["data"] = [(errors, sim_times)]
    error_plotter = spikingFT.utils.plotter.RMSEPlotter(**kwargs)
    error_plotter()
    return

def main():
    single_run_experiment()
    sim_times_experiment()


if __name__ == "__main__":
    main()
