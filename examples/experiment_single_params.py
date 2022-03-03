#!/usr/bin/env python3
"""
Script for testing the SNN class with sample data
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
    norm_data = data - data.min()
    norm_data /= norm_data.max()
    norm_data = (norm_data-0.5) * 2
    ft_np = np.fft.fft(norm_data[0, :]) / (nsamples/2)
    ft_real = ft_np.real[1:int(nsamples/2)]
    ft_imag = ft_np.imag[1:int(nsamples/2)]
    ft_modulus = np.abs(ft_np)[1:int(nsamples/2)]
    return (ft_real, ft_imag, ft_modulus)


def get_sft_components(nsamples, data):
    real_spikes = data[:, 0][1:int(nsamples/2)]
    imag_spikes = data[:, 1][1:int(nsamples/2)]
    sft_real = real_spikes
    sft_imag = imag_spikes
    sft_max = np.max(np.abs(np.hstack([sft_real, sft_imag])))
    sft_real = sft_real / sft_max
    sft_imag = sft_imag / sft_max
    sft_modulus = np.sqrt(sft_real**2 + sft_imag**2)
    sft_modulus = np.log10(9*sft_modulus/sft_modulus.max()+1)
    return (sft_real, sft_imag, sft_modulus)


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


def plot_error(nsamples, data, chirp_n=0):
    """
    Plot relative error histograms
    """
    sft_real, sft_imag, sft_modulus = get_sft_components(nsamples, data)
    ft_real, ft_imag, ft_modulus = get_ft_components(nsamples, data)
    ft_max = np.max(np.abs(np.hstack([ft_real, ft_imag])))
    ft_real = ft_real / ft_max
    ft_imag = ft_imag / ft_max
    ft_modulus = np.sqrt(ft_real**2 + ft_imag**2)

    ft_modulus = np.log10(9*ft_modulus/ft_modulus.max()+1)

    kwargs = {}
    kwargs["plot_names"] = ["real_spectrum", "imag_spectrum", "modulus"]
    kwargs["data"] = [
        (sft_real, ft_real),
        (sft_imag, ft_imag),
        (sft_modulus, ft_modulus)
    ]
    error_plotter = spikingFT.utils.plotter.RelErrorPlotter(**kwargs)
    error_plotter.chirp_n = chirp_n
    fig = error_plotter()
    return fig


def plot_single_chirp(nsamples, data, sim_handler=None, chirp_n=0, plot_spikes=True):
    nsamples = sim_handler.snn.nsamples
    if plot_spikes:
        sim_time = sim_handler.config["snn_config"]["sim_time"]
        spikes = sim_handler.snn.spikes
        voltage = sim_handler.snn.voltage
        plot_simulation(nsamples, sim_time, spikes, voltage)
    fig = plot_error(nsamples,
                     data,
                     chirp_n
                    )
    return fig


def plot_from_files(n_chirps=4, nsamples=1024):
    fft_data = np.load("../data/fft_acc.npy")[:,1:int(nsamples/2)]
    for chirp_n in range(n_chirps):
        sft_data = np.load("./fft_loihi_results/sft_{}.npy".format(chirp_n))
        sft_real, sft_imag, sft_modulus = get_sft_components(nsamples, sft_data)
        fft_real = fft_data[chirp_n].real
        fft_imag = fft_data[chirp_n].imag
        fft_modulus = np.abs(fft_data[chirp_n])
        fft_max = np.max(np.abs(np.hstack([fft_real, fft_imag])))
        fft_real = fft_real / fft_max
        fft_imag = fft_imag / fft_max
        fft_modulus = np.sqrt(fft_real**2 + fft_imag**2)
        fft_modulus = np.log10(9*fft_modulus/fft_modulus.max()+1)


        kwargs = {}
        kwargs["plot_names"] = ["real_spectrum", "imag_spectrum", "modulus"]
        kwargs["data"] = [
            (sft_real, fft_real),
            (sft_imag, fft_imag),
            (sft_modulus, fft_modulus)
        ]
        error_plotter = spikingFT.utils.plotter.RelErrorPlotter(**kwargs)
        error_plotter.chirp_n = chirp_n
        fig = error_plotter()
        plt.show()
        fig.savefig("./fft_loihi_results/error_plot_{}.pdf".format(chirp_n),
                    dpi=150, bbox_inches='tight')
    return


def special_cases(filename="../config/experiment_special_cases.json",
                  snn_from_file=False):
    if not snn_from_file:
        sim_handler = spikingFT.startup.startup(filename, autorun=False)
    else:
        sim_handler = None
    n_chirps = sim_handler.config["data"]["chirps_per_frame"]
    n_samples = sim_handler.config["data"]["samples_per_chirp"]
    platform = sim_handler.config["snn_config"]["framework"]
    mode = sim_handler.config["snn_config"]["mode"]
    figs = []
    folder_path = "./{}_{}_results/".format(mode, platform)
    pathlib.Path(folder_path).mkdir(parents=True, exist_ok=True)
    for chirp_n in range(n_chirps):
        if not snn_from_file:
            sim_handler.run(chirp_n)
            np.save("./{}/sft_{}.npy".format(folder_path, chirp_n),
                    sim_handler.output
            )
        else:
            data = np.load("./fft_loihi_results/sft_{}.npy".format(chirp_n))
        figs.append(plot_single_chirp(n_samples, data, sim_handler, chirp_n, False))
        figs[-1].savefig("./{}/error_plot_{}.pdf".format(folder_path, chirp_n),
                         dpi=150, bbox_inches='tight')
        plt.show()
    return

def multitarget_case(conf_filename="../config/experiment_multiple_targets.json"):
    # Instantiate a simulation handler and run spiking FT with sample data
    sim_handler = spikingFT.startup.startup(conf_filename)
    real_spikes = sim_handler.output[:,0]
    imag_spikes = sim_handler.output[:,1]
    filename = sim_handler.datapath.name
    np.save(filename, real_spikes+1j*imag_spikes)
    return

def main(conf_filename="../config/experiment_special_cases.json"):
    # Instantiate a simulation handler and run spiking FT with sample data
    sim_handler = spikingFT.startup.startup(conf_filename)
    fig = plot_single_chirp(sim_handler.data, sim_handler)
    fig.savefig("./error_plot.pdf", dpi=150, bbox_inches='tight')
    return


if __name__ == "__main__":
    main()
