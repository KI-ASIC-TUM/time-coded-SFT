#!/usr/bin/env python3
"""
Module for loading metadata and starting the simulation
"""
# Standard libraries
import argparse
import json
import logging
import pathlib
import time
# Local libraries
import spikingFT.spiking_ft


def parse_args():
    """
    Obtain the simulation options from the input arguments
    """
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    parser = argparse.ArgumentParser(
        usage="main.py [-h] [-s] config_file")
    parser.add_argument("config_file",
                        type=str,
                        help="Relative location of the configuration file"
                       )
    parser.add_argument("-s",
                        type=str2bool,
                        default=False,
                        nargs='?',
                        const=True,
                        metavar="",
                        help="Show the plot after the simulation"
                       )
    # Get the values from the argument list
    args = parser.parse_args()
    config_file = args.config_file
    show_plot = args.s
    return (config_file, show_plot)


def load_config(conf_file):
    """
    Load the configuration file with the simulation parameters

    @param conf_file: str with the relative address of the config file
    @param dims: Number of Fourier dimensions of the experiment 
    """
    # Load configuaration data from local file
    with open(conf_file) as f:
        config_data = json.load(f)
    fname = config_data["filename"]
    data_path = pathlib.Path(__file__).resolve().parent.joinpath(fname)
    #TODO: Load the configuration parameters
    # config = config_data["cfar_args"]["{}D".format(dims)]
    config = {}
    return data_path, config


def conf_logger():
    # Create log folder
    logpath = pathlib.Path(__file__).resolve().parent.parent.joinpath("log")
    pathlib.Path(logpath).mkdir(parents=True, exist_ok=True)
    datetime = time.strftime("%Y-%m-%d %H:%M:%S")
    fdatetime = time.strftime("%Y%m%d-%H%M%S")
    # Create logger
    logger = logging.getLogger('spiking-FT')
    logger.setLevel(logging.DEBUG)

    # Create file handler
    file_handler = logging.FileHandler("{}/{}.log".format(logpath, fdatetime))
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s',
                                  "%H:%M:%S")
    file_handler.setFormatter(formatter)
    file_handler.stream.write("{} MAIN PROGRAM EXECUTION\n".format(datetime))
    logger.addHandler(file_handler)

    # Create console handler
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(levelname)s: %(message)s')
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def run(data_path, config, show_plot):
    """
    Run the algorithm with the loaded configuration
    """
    spikingFT.spiking_ft.run(data_path, config)
    return


def startup(conf_file, show_plot=True):
    """
    Run the DFT and CFAR on BBM data
    """
    data_path, config = load_config(conf_file)

    logger = conf_logger()
    init_message = "Running spiking-FT:"
    init_message += "\n- Configuration file: {}".format(conf_file)
    logger.info(init_message)

    run(data_path, config, show_plot)
    logger.info("Execution finished")
    return


if __name__ == "__main__":
    conf_file, show_plot = parse_args()
    startup(conf_file, show_plot)
