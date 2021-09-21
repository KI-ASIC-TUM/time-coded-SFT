#!/usr/bin/env python3
"""
Module with functions for generating different experiment plots
"""
# Standard libraries
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt


class Plotter(ABC):
    """
    Abstract class for defining structure of plotters

    Any plot that has to be run in the library shall be created as
    an instance of this class
    """
    def __init__(self, **kwargs):
        """
        Initialize plotter

        Paramters:
            plot_names: List with the names of all plots
            data: List with the data corresponding with the plot names
            show: Bool indicating whether to show the resulting plots
        """
        self.plot_names = kwargs.get("plot_names")
        self.data = kwargs.get("data")
        self.show = kwargs.get("show", True)
        if len(self.plot_names) != len(self.data):
            raise ValueError("Sizes of names and data lists do not match")
        self.nplots = len(self.plot_names)
        self.fig = None
        self.axis = []

    def formatter(self):
        for ax in self.axis:
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
        return

    @abstractmethod
    def plot(self, plot_name, i):
        return

    def run(self):
        self.fig, self.axis = plt.subplots(self.nplots)
        for i, plot_name in enumerate(self.plot_names):
            self.plot(plot_name, i)
        plt.tight_layout()
        self.formatter()
        if self.show:
            plt.show()

    def __call__(self):
        self.run()


class SNNSimulationPlotter(Plotter):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def plot_voltages(self, data, ax):
        nsamples = data.shape[1]
        for sample in range(1, nsamples):
            ax.plot(data[:, sample, 0], linewidth=.5)
            ax.plot(data[:, sample, 1], linewidth=.5)
        ax.set_xlabel("Time step")
        ax.set_ylabel(r'$V_m$ (mV)')
        ax.set_title("Membrane voltages over simulation time")

    def plot_spikes(self, data, ax):
        nsamples = data[0].size
        for sample_n in range(nsamples):
            ax.scatter(data[0][sample_n], sample_n,  s=4, c="#1f77b4")
            ax.scatter(data[0][sample_n], sample_n+nsamples,  s=4, c="#1f77b4")
        ax.set_xlabel("Relative simulation time step")
        ax.set_ylabel("Neuron")
        ax.set_title("Output scatter plot")
        return
    
    def plot_spectrum(self, data, ax):
        ax.plot(data, linewidth=.5)
        ax.set_xlabel("FT bin nº")
        ax.set_ylabel(r'S-FT $n_s$')
        ax.set_title("FT modulus")

    def plot(self, plot_name, plot_n):
        ax = self.axis[plot_n]
        if plot_name == "voltages": 
            self.plot_voltages(self.data[plot_n], ax)
        elif plot_name == "spikes": 
            self.plot_spikes(self.data[plot_n], ax)
        elif plot_name == "FT": 
            self.plot_spectrum(self.data[plot_n], ax)
        else:
            raise ValueError("Invalid plot name: {}".format(plot_name))


class RelErrorPlotter(Plotter):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def plot_component(self, data, ax_left, component="", legend=False):
        ax_right = ax_left.twinx()
        l1 = ax_left.plot(data[0], color="blue", label="signal", linewidth=.5)
        l2 = ax_right.plot(data[1], color="red", linestyle="--", label="error", linewidth=.5)
        ax_right.set_ylim([0., 0.05])
        ax_left.set_ylabel("FT")
        ax_right.set_ylabel("Rel. error")
        lines = l1 + l2
        labels = [l.get_label() for l in lines]
        if legend:
            ax_left.legend(lines, labels, loc='upper right')
        ax_left.set_title("{} spectrum rel. error".format(component))
        ax_right.spines['top'].set_visible(False)

    def plot(self, plot_name, plot_n):
        ax = self.axis[plot_n]
        if plot_name == "real_spectrum":
            self.plot_component(self.data[plot_n], ax, "Real", True)
        elif plot_name == "imag_spectrum":
            self.plot_component(self.data[plot_n], ax, "Imaginary")
        elif plot_name == "modulus":
            self.plot_component(self.data[plot_n], ax, "Absolute")
        else:
            raise ValueError("Invalid plot name: {}".format(plot_name))
