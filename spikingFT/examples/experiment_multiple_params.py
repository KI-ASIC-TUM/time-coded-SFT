#!/usr/bin/env python3
"""
Script for testing the SNNNumpy class with sample data
"""
# Standard libraries
import logging
import matplotlib.pyplot as plt
import numpy as np
import pathlib
# Local libraries
import run_sft
import spikingFT.startup
import spikingFT.utils. plotter

logger = logging.getLogger('spiking-FT')


def iterate_over_samples(sim_handler, sim_times):
    """
    Iterate over the different sim times and simulate network
    """
    errors = []
    for sim_time in sim_times:
        # Update handler configuration
        sim_handler.config["snn_config"]["sim_time"] = sim_time
        sim_handler.run()
        errors.append(sim_handler.metrics["rmse"])
    return errors


def single_nsamples(sim_handler):
    # Get simulation times from config file
    exp_config = sim_handler.config["experiment"]
    sim_times = exp_config["sim_times"]
    time_step = sim_handler.config["snn_config"]["time_step"]
    # Iterate over the different sim times
    errors = iterate_over_samples(sim_handler, sim_times)
    # Plot results
    nsteps = [sim_time/time_step for sim_time in sim_times]
    kwargs = {}
    kwargs["plot_names"] = ["single_line"]
    kwargs["data"] = [(errors, nsteps)]
    error_plotter = spikingFT.utils.plotter.RMSEPlotter(**kwargs)
    fig = error_plotter()
    folder_path = run_sft.load_path(sim_handler)
    fig.savefig("{}/single_rmse.pdf".format(folder_path), dpi=150)
    logger.info("Figure saved in {}".format(folder_path))


def multiple_nsamples(sim_handler):
    # Get simulation times from config file
    exp_config = sim_handler.config["experiment"]
    sim_times = exp_config["sim_times"]
    time_step = sim_handler.config["snn_config"]["time_step"]
    samples_per_chirp = exp_config["samples_per_chirp"]
    errors = []
    # Iterate over the different number of samples
    for nsamples in samples_per_chirp:
        # Update handler configuration
        sim_handler.config["data"]["samples_per_chirp"] = nsamples
        errors.append(iterate_over_samples(sim_handler, sim_times))
    # Plot results
    nsteps = [sim_time/time_step for sim_time in sim_times]
    kwargs = {}
    kwargs["plot_names"] = ["multiple_lines"]
    kwargs["data"] = [(errors, nsteps, samples_per_chirp)]
    error_plotter = spikingFT.utils.plotter.RMSEPlotter(**kwargs)
    fig = error_plotter()
    folder_path = run_sft.load_path(sim_handler)
    fig.savefig("{}/rmse_plot_multi.pdf".format(folder_path), dpi=150)
    logger.info("Figure saved in {}".format(folder_path))


def main(conf_filename):
    # Instantiate a simulation handler with specified configuration
    sim_handler = spikingFT.startup.startup(conf_filename, autorun=False)
    # single_nsamples(sim_handler)
    multiple_nsamples(sim_handler)
    return


if __name__ == "__main__":
    main_path = pathlib.Path(__file__).resolve().parent.parent
    conf_path = main_path.joinpath("config/experiment_simtimes.json")
    main(conf_path)
