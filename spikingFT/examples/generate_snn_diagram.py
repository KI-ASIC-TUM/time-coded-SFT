#!/usr/bin/env python3
"""
Script for generating sample diagram of the S-FT

The plotting library is tuned for the specific radar chirp that is
indicated in the default configuration file. If this data is changed,
the plotting function should be changed accordingly.
"""
# Standard libraries
import logging
import matplotlib.pyplot as plt
import numpy as np
import pathlib
# Local libraries
import run_sft
import spikingFT.startup
import spikingFT.utils.plotter
import spikingFT.utils.metrics

logger = logging.getLogger('spiking-FT')


def main(conf_filename):
    # Instantiate a simulation handler and run spiking FT with sample data
    sim_handler = spikingFT.startup.startup(conf_filename, autorun=True)
    voltage = sim_handler.snn.voltage[:, 1, 0]
    fig = spikingFT.utils.plotter.plot_snn_diagram(
        voltage,
        sim_handler.snn.v_threshold*1.1,
        voltage.max(),
        show=True
    )
    folder_path = run_sft.load_path(sim_handler)
    fig.savefig("{}/snn_membrane_diagram.pdf".format(folder_path),
                dpi=150,
                bbox_inches='tight'
               )
    logger.info("Figure saved in {}".format(folder_path))
    return


if __name__ == "__main__":
    main_path = pathlib.Path(__file__).resolve().parent.parent
    conf_path = main_path.joinpath("config/generate_snn_diagram.json")
    main(conf_path)
