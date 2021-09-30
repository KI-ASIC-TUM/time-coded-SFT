#!/usr/bin/env python3
"""
Script for testing the SNNNumpy class with sample data
"""
# Standard libraries
import matplotlib.pyplot as plt
import numpy as np
import pathlib
# Local libraries
import spikingFT.startup
import spikingFT.utils.plotter
import spikingFT.utils.metrics


def get_ft_components(nsamples, data):
    """
    Return the real imaginary, and modulus of the np.fft of input data
    """
    ft_np = np.fft.fft(data[0, :])
    ft_np_comps = np.vstack((ft_np.real, ft_np.imag)).transpose()
    ft_np_norm = spikingFT.utils.metrics.simplify_ft(ft_np_comps)
    ft_real = ft_np_norm[:, 0]
    ft_imag = ft_np_norm[:, 1]
    ft_modulus = np.abs(ft_np)[1:int(nsamples/2)]
    ft_modulus -= ft_modulus.min()
    ft_modulus /= ft_modulus.max()
    return (ft_real, ft_imag, ft_modulus)


def plot_simulation(nsamples, sim_time, spikes, voltage):
    real_spikes = spikes[:, 0][1:int(nsamples/2)]
    imag_spikes = spikes[:, 1][1:int(nsamples/2)]

    # Plot S-FT result and reference result
    kwargs = {}
    kwargs["plot_names"] = ["voltages", "spikes"]
    kwargs["data"] = [
        voltage,
        (real_spikes, imag_spikes, 2*sim_time),
    ]
    sim_plotter = spikingFT.utils.plotter.SNNSimulationPlotter(**kwargs)
    fig = sim_plotter()
    fig.savefig("./simulation_plot.pdf", dpi=150, bbox_inches='tight')


def plot_error(nsamples, data, output, spikes, rel_error):
    """
    Plot relative error histograms
    """
    spikes = spikingFT.utils.metrics.simplify_ft(spikes[:, :, -1])
    real_spikes = output[:, 0][1:int(nsamples/2)]
    imag_spikes = output[:, 1][1:int(nsamples/2)]
    sft_modulus = np.sqrt(real_spikes**2 + imag_spikes**2)
    sft_modulus -= sft_modulus.min()
    sft_modulus /= sft_modulus.max()
    ft_real, ft_imag, ft_modulus = get_ft_components(nsamples, data)

    real_spikes -= real_spikes.min()
    real_spikes /= real_spikes.max()
    # real_spikes = 1 - real_spikes
    imag_spikes -= imag_spikes.min()
    imag_spikes /= imag_spikes.max()
    # imag_spikes = 1 - imag_spikes

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
    error_plotter = spikingFT.utils.plotter.RelErrorPlotter(**kwargs)
    fig = error_plotter()
    return fig


def plot_single_chirp(sim_handler, plot_spikes=True):
    nsamples = sim_handler.snn.nsamples
    sim_time = sim_handler.config["snn_config"]["sim_time"]
    spikes = sim_handler.snn.spikes
    voltage = sim_handler.snn.voltage
    if plot_spikes:
        plot_simulation(nsamples, sim_time, spikes, voltage)
    rel_error = sim_handler.metrics["rel_error"]
    fig = plot_error(nsamples, sim_handler.data, sim_handler.output, spikes, rel_error)
    return fig


def special_cases(filename="../config/experiment_special_cases.json"):
    sim_handler = spikingFT.startup.startup(filename, autorun=False)
    n_chirps = sim_handler.config["data"]["chirps_per_frame"]
    platform = sim_handler.config["snn_config"]["framework"]
    mode = sim_handler.config["snn_config"]["mode"]
    figs = []
    folder_path = "./{}_{}_results/".format(mode, platform)
    pathlib.Path(folder_path).mkdir(parents=True, exist_ok=True)
    for chirp_n in range(n_chirps):
        sim_handler.run(chirp_n)
        figs.append(plot_single_chirp(sim_handler, False))
        figs[-1].savefig("./{}/error_plot_{}.pdf".format(folder_path, chirp_n),
                         dpi=150, bbox_inches='tight')
    return


def main(conf_filename="../config/test_experiment.json"):
    # Instantiate a simulation handler and run spiking FT with sample data
    sim_handler = spikingFT.startup.startup(conf_filename)
    fig = plot_single_chirp(sim_handler)
    fig.savefig("./error_plot.pdf", dpi=150, bbox_inches='tight')
    return


if __name__ == "__main__":
    special_cases()
