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


def main(conf_filename="../config/test_experiment.json"):
    # Instantiate a simulation handler and run spiking FT with sample data
    sim_handler = spikingFT.startup.startup(conf_filename)
    nsamples = sim_handler.snn.nsamples
    sim_time = sim_handler.config["snn_config"]["sim_time"]
    real_spikes = sim_handler.snn.spikes[:, 0][1:int(nsamples/2)]
    imag_spikes = sim_handler.snn.spikes[:, 1][1:int(nsamples/2)]
    real_spikes_norm = sim_handler.output[:, 0][1:int(nsamples/2)]
    imag_spikes_norm = sim_handler.output[:, 1][1:int(nsamples/2)]

    input_spikes = sim_handler.encoded_data.real
    # Split input spikes in two subgroups,
    # so it has the same format as an SNN layer
    input_spikes_1 = input_spikes[0, :int(nsamples/2)]
    input_spikes_2 = input_spikes[0, int(nsamples/2):]

    # Plot S-FT result and reference result
    kwargs = {}
    kwargs["plot_names"] = ["spikes", "voltages", "spikes"]
    kwargs["data"] = [
        (input_spikes_1, input_spikes_2, 2*sim_time),
        sim_handler.snn.voltage,
        (real_spikes, imag_spikes, 2*sim_time),
    ]
    kwargs["sim_time"] = sim_time
    sim_plotter = spikingFT.utils.plotter.SNNLayersPlotter(**kwargs)
    fig = sim_plotter()
    fig.savefig("./simulation_plot.pdf", dpi=150, bbox_inches='tight')
    return


if __name__ == "__main__":
    main()
