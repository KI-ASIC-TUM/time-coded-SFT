#!/usr/bin/env python3
"""
Script for testing the SNNNumpy class with sample data
"""
# Standard libraries
import matplotlib.pyplot as plt
import numpy as np
# Local libraries
import spikingFT.startup
import spikingFT.utils.plotter


def main(conf_filename="../config/test_experiment_brian.json"):
    # Instantiate a simulation handler and run spiking FT with sample data
    sim_handler = spikingFT.startup.startup(conf_filename)
    nsamples = sim_handler.snn.nsamples
    nlayers = sim_handler.snn.nlayers
    sim_time = sim_handler.config["snn_config"]["sim_time"]
    time_step = sim_handler.config["snn_config"]["time_step"]
    total_time = sim_time * (nlayers+1)
    real_spikes = sim_handler.snn.spikes[:, 0][1:int(nsamples), :]
    imag_spikes = sim_handler.snn.spikes[:, 1][1:int(nsamples), :]

    # Split input spikes in two subgroups,
    # so it has the same format as an SNN layer
    input_spikes = sim_handler.encoded_data.real
    input_spikes_1 = input_spikes[0, :int(nsamples/2)]
    input_spikes_2 = input_spikes[0, int(nsamples/2):]

    # Plot S-FT result and reference result
    kwargs = {}
    kwargs["plot_names"] = ["spikes"]
    kwargs["data"] = [(input_spikes_1, input_spikes_2, total_time)]
    for n in range(nlayers):
        kwargs["plot_names"].append("voltages")
        kwargs["plot_names"].append("spikes")
        kwargs["data"].append(sim_handler.snn.voltage[:, :, :, n])
        kwargs["data"].append(
            (real_spikes[:, n], imag_spikes[:, n], total_time)
        )
    kwargs["sim_time"] = sim_time
    kwargs["tight_layout"] = False
    kwargs["nlayers"] = nlayers
    kwargs["time_step"] = time_step
    sim_plotter = spikingFT.utils.plotter.SNNLayersPlotter(**kwargs)
    fig = sim_plotter()
    fig.savefig("./simulation_plot.pdf", dpi=150, bbox_inches='tight')
    return


if __name__ == "__main__":
    main()
