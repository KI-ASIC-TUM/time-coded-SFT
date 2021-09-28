#!/usr/bin/env python3
"""
Module with functions for generating different experiment plots
"""
# Standard libraries
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import numpy as np


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
        plt.rcParams['font.size'] = 20
        #    plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['axes.labelsize'] = 0.5*plt.rcParams['font.size']
        plt.rcParams['axes.titlesize'] = plt.rcParams['font.size']
        plt.rcParams['legend.fontsize'] = 0.5*plt.rcParams['font.size']
        plt.rcParams['xtick.labelsize'] = 0.4*plt.rcParams['font.size']
        plt.rcParams['ytick.labelsize'] = 0.4*plt.rcParams['font.size']
        plt.rcParams['xtick.major.size'] = 3
        plt.rcParams['xtick.minor.size'] = 3
        plt.rcParams['xtick.major.width'] = 1
        plt.rcParams['xtick.minor.width'] = 1
        plt.rcParams['ytick.major.size'] = 3
        plt.rcParams['ytick.minor.size'] = 3
        plt.rcParams['ytick.major.width'] = 1
        plt.rcParams['ytick.minor.width'] = 1
        plt.rcParams['legend.frameon'] = True
        plt.rcParams['legend.loc'] = 'upper right'
        plt.rcParams['axes.linewidth'] = 1
        plt.rcParams['lines.linewidth'] = 1
        plt.rcParams['lines.markersize'] = 3
        return

    @abstractmethod
    def plot(self, plot_name, i):
        return

    def run(self):
        self.fig, self.axis = plt.subplots(self.nplots)
        if type(self.axis) is not np.ndarray:
            self.axis = np.array(self.axis).reshape((1,))
        for i, plot_name in enumerate(self.plot_names):
            self.plot(plot_name, i)
        self.formatter()
        plt.tight_layout()
        if self.show:
            plt.show()
        return self.fig

    def __call__(self):
        return self.run()


class SNNSimulationPlotter(Plotter):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def plot_voltages(self, data, ax):
        nsamples = data.shape[1]
        for sample in range(1, nsamples):
            ax.plot(data[:, sample, 0], linewidth=.5)
            ax.plot(data[:, sample, 1], linewidth=.5)
        ax.set_xlabel("Time step")
        # ax.set_ylabel(r'$V_m$ (mV)')
        ax.spines['left'].set_visible(False)
        ax.set_yticks([])
        ax.set_title("Membrane voltages over simulation time")

    def plot_spikes(self, data, ax):
        nsamples = data[0].size
        for sample_n in range(nsamples):
            ax.scatter(data[0][sample_n], sample_n,  s=4, c="#1f77b4")
            ax.scatter(data[1][sample_n], sample_n+nsamples,  s=4, c="#1f77b4")
        ax.set_xlim(0, data[2])
        ax.set_xlabel("simulation time")
        # ax.set_ylabel("Neuron")
        ax.spines['left'].set_visible(False)
        ax.set_yticks([])
        ax.set_title("Output scatter plot")
        return
    
    def plot_spectrum(self, data, ax):
        ax.plot(data[0], linewidth=.5)
        ax.plot(data[1], linewidth=.5)
        ax.set_xlabel("FT bin nº")
        # ax.set_ylabel(r'S-FT $t_s$')
        ax.spines['left'].set_visible(False)
        ax.set_yticks([])
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


class SNNLayersPlotter(Plotter):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def plot_voltages(self, data, ax):
        nsamples = data.shape[1]
        for sample in range(1, nsamples):
            t = np.linspace(0, data.shape[0], data.shape[0])
            ax.plot(t, data[:, sample, 0], linewidth=.5)
            ax.plot(t, data[:, sample, 1], linewidth=.5)
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_xlim(xmin=0)
        ax.set_ylabel(r'${v_m}$')
        ax.spines['left'].set_position('zero')
        ax.spines['bottom'].set_position('zero')
        ax.plot((1), (0), ls="", marker=">", ms=2, color="k",
            transform=ax.get_yaxis_transform(), clip_on=False)
        ax.plot((0), (1), ls="", marker="^", ms=2, color="k",
                transform=ax.get_xaxis_transform(), clip_on=False)

    def plot_spikes(self, data, ax, color="#1f77b4"):
        nsamples = data[0].size
        for sample_n in range(nsamples):
            ax.scatter(data[0][sample_n], sample_n,  s=4, c=color)
            ax.scatter(data[1][sample_n], sample_n+nsamples,  s=4, c=color)
        ax.set_xlim(0, data[-1])
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_ylabel(r'$t_s$')
        ax.spines['left'].set_position('zero')
        ax.spines['bottom'].set_position('zero')
        ax.plot((1), (0), ls="", marker=">", ms=2, color="k",
            transform=ax.get_yaxis_transform(), clip_on=False)
        ax.plot((0), (1), ls="", marker="^", ms=2, color="k",
                transform=ax.get_xaxis_transform(), clip_on=False)

    def plot(self, plot_name, plot_n):
        ax = self.axis[plot_n]
        if plot_name == "voltages":
            self.plot_voltages(self.data[plot_n], ax)
        elif plot_name == "spikes":
            if plot_n == 0:
                color = "red"
            else:
                color = "#1f77b4"
            self.plot_spikes(self.data[plot_n], ax, color)
        else:
            raise ValueError("Invalid plot name: {}".format(plot_name))


class RelErrorPlotter(Plotter):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def plot_component(self, data, ax_left, component="", legend=False, xlabel=False):
        ax_right = ax_left.twinx()
        l1 = ax_left.plot(data[0], color="blue", label="signal", linewidth=.5)
        ax_left.plot(data[1], color="orange", label="signal", linewidth=.5)
        l2 = ax_right.plot(data[-1], color="red", linestyle="--", label="error", linewidth=.5)
        ax_right.set_ylim([0., 0.1])
        ax_left.set_ylim([0., 1])
        ax_left.set_ylabel("FT")
        ax_right.set_ylabel(r'$\varepsilon$')
        lines = l1 + l2
        labels = [l.get_label() for l in lines]
        if legend:
            ax_left.legend(lines, labels, loc='upper right')
        if xlabel:
            ax_left.set_xlabel("Bin Nº")
        ax_left.set_title("{} spectrum rel. error".format(component))
        for ax in (ax_right, ax_left):
            ax.spines['top'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.tick_params(axis="y", which="both",length=0)
        # Align right and left ticks
        align_yaxes([ax_left, ax_right])
        ax_left.grid(axis='y')

    def plot(self, plot_name, plot_n):
        ax = self.axis[plot_n]
        if plot_name == "real_spectrum":
            self.plot_component(self.data[plot_n], ax, "Real", legend=True)
        elif plot_name == "imag_spectrum":
            self.plot_component(self.data[plot_n], ax, "Imaginary")
        elif plot_name == "modulus":
            self.plot_component(self.data[plot_n], ax, "Absolute", xlabel=True)
        else:
            raise ValueError("Invalid plot name: {}".format(plot_name))


class RMSEPlotter(Plotter):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def plot_single_line(self, data, ax):
        """
        Plot RMSE evolution for a single nsamples batch
        """
        ax.plot(data[1], data[0], c="red")
        ax.set_title("RMSE over different sim times")
        ax.set_ylabel("RMSE")
        ax.set_xlabel("Nº simulation steps")
        ax.spines['left'].set_visible(False)
        ax.grid(axis='y')
        ax.tick_params(axis="y", which="both",length=0)
        ax.set_ylim(0, 0.05)
        ax.locator_params(axis='x', nbins=5)
        return

    def plot_multiple_lines(self, data, ax):
        """
        Plot RMSE evolution for a single nsamples batch
        """
        for i, data_stream in enumerate(data[0]):
            tsteps = data[1]
            nbins = data[2][i]
            ax.plot(tsteps, data_stream, label="{} bins".format(nbins))
        ax.set_title("RMSE over different sim times")
        ax.set_ylabel("RMSE")
        ax.set_xlabel("Nº simulation steps")
        ax.spines['left'].set_visible(False)
        ax.grid(axis='y')
        ax.tick_params(axis="y", which="both",length=0)
        ax.set_ylim(0, 0.05)
        ax.locator_params(axis='x', nbins=5)
        ax.legend()
        return

    def plot(self, plot_name, plot_n):
        ax = self.axis[plot_n]
        if plot_name == "single_line":
            self.plot_single_line(self.data[plot_n], ax)
        elif plot_name == "multiple_lines":
            self.plot_multiple_lines(self.data[plot_n], ax)
        else:
            raise ValueError("Invalid plot name: {}".format(plot_name))


def align_yaxes(axes, nbins=3):
    """
    Align the ticks of multiple y axes

    Args:
        axes (list): list of axes objects whose yaxis ticks are to be aligned.
    Returns:
        new_ticks (list): a list of new ticks for each axis in <axes>.
        A new sets of ticks are computed for each axis in <axes> but with equal
        length.
    """
    nax = len(axes)
    ticks = [aii.get_yticks() for aii in axes]
    aligns = [ticks[ii][0] for ii in range(nax)]
    bounds = [aii.get_ylim() for aii in axes]
    # align at some points
    ticks_align = [ticks[ii]-aligns[ii] for ii in range(nax)]
    # scale the range to 1-100
    ranges = [tii[-1]-tii[0] for tii in ticks]
    lgs = [-np.log10(rii)+2. for rii in ranges]
    igs = [np.floor(ii) for ii in lgs]
    log_ticks = [ticks_align[ii]*(10.**igs[ii]) for ii in range(nax)]
    # put all axes ticks into a single array, then compute new ticks for all
    comb_ticks = np.concatenate(log_ticks)
    comb_ticks.sort()
    locator = plt.MaxNLocator(nbins=nbins, steps=[1, 2, 2.5, 3, 4, 5, 8, 10])
    new_ticks = locator.tick_values(comb_ticks[0], comb_ticks[-1])
    new_ticks = [new_ticks/10.**igs[ii] for ii in range(nax)]
    new_ticks = [new_ticks[ii]+aligns[ii] for ii in range(nax)]
    # find the lower bound
    idx_l = 0
    for i in range(len(new_ticks[0])):
        if any([new_ticks[jj][i] > bounds[jj][0] for jj in range(nax)]):
            idx_l = i-1
            break
    # find the upper bound
    idx_r = 0
    for i in range(len(new_ticks[0])):
        if all([new_ticks[jj][i] > bounds[jj][1] for jj in range(nax)]):
            idx_r = i
            break
    # trim tick lists by bounds
    new_ticks = [tii[idx_l:idx_r+1] for tii in new_ticks]
    # set ticks for each axis
    for axii, tii in zip(axes, new_ticks):
        axii.set_yticks(tii)
    return new_ticks
