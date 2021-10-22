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
import spikingFT.utils.metrics


def main(conf_filename="../config/test_experiment.json"):
    # Instantiate a simulation handler and run spiking FT with sample data
    sim_handler = spikingFT.startup.startup(conf_filename, autorun=True)
    voltage = sim_handler.snn.voltage[:, 3, 0]
    fig = spikingFT.utils.plotter.plot_snn_diagram(
        voltage,
        sim_handler.snn.v_threshold*1.1,
        voltage.max(),
        show=True
    )
    fig.savefig("./snn_membrane_diagram.pdf", dpi=150, bbox_inches='tight')
    return


if __name__ == "__main__":
    main()
