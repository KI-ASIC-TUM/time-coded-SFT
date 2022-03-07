#!/usr/bin/env python3
"""
Module containing a parser for the arguments in the terminal execution

This module is meant to be used by the startup.py file, as well as the
examples present in the examples folder
"""
# Standard libraries
import argparse
# Local libraries


def parse_args():
    """
    Obtain the simulation options from the input arguments

    Return a tuple with the following elements:
    * config_file: Relative location of the simulation configuration file
    * show_plot: If true, plots shall be generated after the simulation
    * from_file: If true, the output of the SNN is taken from a local
    file that was generated on a previous simulation
    """
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('', 'no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    parser = argparse.ArgumentParser(
        usage="main.py [-h] [-s] [-f] [-c, --config config_file]")
    parser.add_argument("-c", "--config",
                        type=str,
                        required=True,
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
    parser.add_argument("-f",
                        type=str2bool,
                        default=False,
                        nargs='?',
                        const=True,
                        metavar="",
                        help="Fetch the output data from local file, "
                             "instead of running the SNN"
                       )
    # Get the values from the argument list
    args = parser.parse_args()
    config_file = args.config
    show_plot = args.s
    from_file = args.f
    return (config_file, show_plot, from_file)
