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


def main(conf_filename="../config/test_experiment_simtimes.json"):
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


if __name__ == "__main__":
    main()
