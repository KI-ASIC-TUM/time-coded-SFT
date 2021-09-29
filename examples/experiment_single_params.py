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


def plot_single_chirp(sim_handler, plot_spikes=True):
    nsamples = sim_handler.snn.nsamples
    sim_time = sim_handler.config["snn_config"]["sim_time"]
    spikes = spikingFT.utils.metrics.simplify_ft(sim_handler.snn.spikes)
    real_spikes = 1 - spikes[:, 0]
    imag_spikes = 1 - spikes[:, 1]
    real_spikes_norm = sim_handler.output[:, 0][1:int(nsamples/2)]
    imag_spikes_norm = sim_handler.output[:, 1][1:int(nsamples/2)]
    sft_modulus = np.sqrt(real_spikes_norm**2 + imag_spikes_norm**2)
    sft_modulus -= sft_modulus.min()
    sft_modulus /= sft_modulus.max()
    ft_np = np.fft.fft(sim_handler.data[0, :])
    ft_np_comps = np.vstack((ft_np.real, ft_np.imag)).transpose()
    ft_np_norm = spikingFT.utils.metrics.simplify_ft(ft_np_comps)
    ft_real = ft_np_norm[:, 0]
    ft_imag = ft_np_norm[:, 1]
    ft_modulus = np.abs(ft_np)[1:int(nsamples/2)]
    ft_modulus -= ft_modulus.min()
    ft_modulus /= ft_modulus.max()


    if plot_spikes:
        # Plot S-FT result and reference result
        kwargs = {}
        kwargs["plot_names"] = ["voltages", "spikes"]
        kwargs["data"] = [
            sim_handler.snn.voltage,
            (real_spikes, imag_spikes, 2*sim_time),
        ]
        sim_plotter = spikingFT.utils.plotter.SNNSimulationPlotter(**kwargs)
        fig = sim_plotter()
        fig.savefig("./simulation_plot.pdf", dpi=150, bbox_inches='tight')

    # Plot relative error histograms
    rel_error = sim_handler.metrics["rel_error"]
    real_error = rel_error[:, 0]
    imag_error = rel_error[:, 1]
    abs_error = (real_error + imag_error) / 2
    kwargs = {}
    kwargs["plot_names"] = ["real_spectrum", "imag_spectrum", "modulus"]
    kwargs["data"] = [
        (real_spikes, ft_real, real_error),
        (imag_spikes, ft_imag, imag_error),
        (sft_modulus, ft_modulus, abs_error)
    ]
    error_plotter = spikingFT.utils.plotter.RelErrorPlotter(**kwargs, show=False)
    fig = error_plotter()
    return fig


def special_cases(filename="../config/experiment_special_cases.json"):
    sim_handler = spikingFT.startup.startup(filename, autorun=False)
    n_chirps = sim_handler.config["data"]["chirps_per_frame"]
    platform = sim_handler.config["snn_config"]["framework"]
    figs = []
    for chirp_n in range(n_chirps):
        sim_handler.run(chirp_n)
        figs.append(plot_single_chirp(sim_handler, False))
        figs[-1].savefig("./{}_results/error_plot_{}.pdf".format(platform, chirp_n),
                         dpi=150, bbox_inches='tight')
    return


def main(conf_filename="../config/test_experiment_loihi.json"):
    # Instantiate a simulation handler and run spiking FT with sample data
    sim_handler = spikingFT.startup.startup(conf_filename)
    fig = plot_single_chirp(sim_handler)
    fig.savefig("./error_plot.pdf", dpi=150, bbox_inches='tight')
    return


if __name__ == "__main__":
    special_cases()
