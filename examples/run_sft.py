#!/usr/bin/env python3
"""
Run S-FT over FMCW individual chirps
"""
# Standard libraries
import logging
import matplotlib.pyplot as plt
import numpy as np
import pathlib
# Local libraries
import spikingFT.startup
import spikingFT.utils.metrics
import spikingFT.utils.plotter
import spikingFT.utils.parse_args

logger = logging.getLogger('spiking-FT')


def load_path(sim_handler):
    """
    Return simulation folder path
    """
    platform = sim_handler.config["snn_config"]["framework"]
    mode = sim_handler.config["snn_config"]["mode"]
    # Get path to results folder
    folder_path = "{}_{}_results/".format(mode, platform)
    pathlib.Path(folder_path).mkdir(parents=True, exist_ok=True)
    return folder_path


def run_sft(sim_handler, path):
    """
    Run S-FT with the specified configuration
    """
    n_chirps = sim_handler.config["data"]["chirps_per_frame"]
    n_samples = sim_handler.config["data"]["samples_per_chirp"]
    # Run S-FT
    output = np.zeros((n_chirps, n_samples), dtype=np.complex)
    for chirp_n in range(n_chirps):
        sim_handler.run(chirp_n)
        real_spikes = sim_handler.output[:, 0]
        imag_spikes = sim_handler.output[:, 1]
        output[chirp_n, :] = real_spikes + 1j*imag_spikes
    # Save results to local file
    np.save("{}/sft.npy".format(path), output)
    logger.info("SFT results saved in {}".format(path))
    return


def run_fft_np(raw_data, nsamples):
    """
    Run FFT from NumPy on data and save it to local file
    """
    fft_np = np.fft.fft(raw_data[0, :, :])
    np.save("./fft/fft_np_{}".format(nsamples), fft_np)
    logger.info("NumPy results saved in fft/")
    return


def get_complex_terms(ft_data, nsamples):
    """
    Split input data into real and imaginary components
    """
    ft_real = ft_data.real[:, 1:int(nsamples/2)]
    ft_imag = ft_data.imag[:, 1:int(nsamples/2)]
    return ft_real, ft_imag


def normalize(ft_real, ft_imag):
    """
    normalize FT spectrum to a maximum of 1
    """
    # Divide by global maximum
    max_value = np.max((np.abs(ft_real).max(axis=1),
                        np.abs(ft_imag).max(axis=1)),
                        axis=0).reshape((-1, 1))
    ft_real_norm = ft_real / max_value
    ft_imag_norm = ft_imag / max_value
    return (ft_real_norm, ft_imag_norm)


def get_modulus(ft_real, ft_imag):
    ft_modulus = np.sqrt(ft_real**2 + ft_imag**2)
    ft_modulus = np.log10(9*ft_modulus/ft_modulus.max()+1)
    return ft_modulus


def adjust_ft_data(sft_results, fft_results):
    """
    Re-scale and crop data for plotting purposes
    """
    assert(sft_results.shape == fft_results.shape)
    n_chirps, nsamples = sft_results.shape
    # Get real and imaginary components
    # Crop half of the spectrum and remove offset bin
    sft_real, sft_imag = get_complex_terms(sft_results, nsamples)
    fft_real, fft_imag = get_complex_terms(fft_results, nsamples)
    # Normalize all data between 0 and 1
    sft_real_norm, sft_imag_norm = normalize(sft_real, sft_imag)
    fft_real_norm, fft_imag_norm = normalize(fft_real, fft_imag)
    # Get modulus of FT
    sft_modulus = get_modulus(sft_real_norm, sft_imag_norm)
    fft_modulus = get_modulus(fft_real_norm, fft_imag_norm)
    # Merge all data in a list
    adjusted_data = []
    for chirp_n in range(n_chirps):
        adjusted_data.append([
            (sft_real_norm[chirp_n, :], fft_real_norm[chirp_n, :]),
            (sft_imag_norm[chirp_n, :], fft_imag_norm[chirp_n, :]),
            (sft_modulus[chirp_n, :], fft_modulus[chirp_n, :])
        ])
    return adjusted_data


def plot_chirp(ft_data, chirp_n, comparison_source):
    kwargs = {}
    kwargs["plot_names"] = ["real_spectrum", "imag_spectrum", "modulus"]
    kwargs["data"] = ft_data
    kwargs["source"] = comparison_source
    error_plotter = spikingFT.utils.plotter.RelErrorPlotter(**kwargs)
    error_plotter.chirp_n = chirp_n
    fig = error_plotter()
    return fig


def plot_results(path, n_samples, n_chirps, comparison_source="numpy"):
    """
    Plot the results of the S-FT

    @param path: str with the path to S-FT results
    @param n_samples: int representing number of samples per chirp
    @param n_chirps: int representing number of chirps to be analyzed
    @param comparison_source: str indicating which source to use for
     comparing the S-FT with (numpy | acc)
    """
    sft_results = np.load("./{}/sft.npy".format(path))
    if comparison_source == "numpy":
        fft_results = np.load("./fft/fft_np_{}.npy".format(n_samples))
    elif comparison_source == "acc":
        fft_results = np.load("./fft/fft_acc_{}.npy".format(n_samples))
    else:
        raise ValueError("The comparison source is not valid")
    adjusted_data = adjust_ft_data(sft_results, fft_results)
    for chirp_n in range(n_chirps):
        fig = plot_chirp(adjusted_data[chirp_n], chirp_n, comparison_source)
        fig.savefig("{}/error_plot_{}.pdf".format(path, chirp_n),
                    dpi=150, bbox_inches='tight')
        plt.show()
    logger.info("Figures saved in {}".format(path))
    return


def main(conf_filename, from_file, plot):
    """
    @param conf_filename: str with the path to the SNN configuration
    @param from_file: bool indicating whether to run or not the s-ft.
      if True, the s-ft output is read from a local file
    @param plot: bool for indicating whether to plot results
    """
    # Load simulation handler and data parameters
    sim_handler = spikingFT.startup.startup(conf_filename, autorun=False)
    results_path = load_path(sim_handler)
    n_samples = sim_handler.config["data"]["samples_per_chirp"]
    n_chirps = sim_handler.config["data"]["chirps_per_frame"]
    # Instantiate and run S-FT
    if not from_file:
        run_sft(sim_handler, results_path)
        run_fft_np(sim_handler.raw_data, n_samples)
    # Plot results
    if plot:
        plot_results(results_path, n_samples, n_chirps)
    return


if __name__ == "__main__":
    fname, show_plot, from_file = spikingFT.utils.parse_args.parse_args()
    main(fname, from_file, show_plot)
