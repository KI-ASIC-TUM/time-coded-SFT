#!/usr/bin/env python3
"""
Script for generating sample diagram of the S-FT

The plotting library is tuned for the specific radar chirp that is
indicated in the default configuration file. If this data is changed,
the plotting function should be changed accordingly.
"""
# Standard libraries
import matplotlib.pyplot as plt
import numpy as np
# Local libraries
import spikingFT.startup
import spikingFT.utils.plotter
import spikingFT.utils.metrics


def main(conf_filename="../config/generate_snn_diagram.json"):
    # Instantiate a simulation handler and run spiking FT with sample data
    sim_handler = spikingFT.startup.startup(conf_filename, autorun=True)
    voltage = sim_handler.snn.voltage[:, 1, 0]
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
