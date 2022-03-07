
# Spiking FT

Implementation of a Spiking Neural Network (SNN) for solving the Fourier transform (FT) in the neuromorphic chip Loihi.
The project includes an SNN for solving the DFT and another SNN for solving the FFT.
The code can be tested on automotive FMCW radar data corresponding to 1 dynamic and 4 static real-world scenarios.

The neuron model and conducted experiments are described in [the following paper](https://arxiv.org/abs/2202.12650):

_López-Randulfe, J., Reeb, N., Karimi, N., Liu, C., Gonzalez, H., Dietrich, R., Vogginger, B.,
Mayr, C., and Knoll, A. "Time-coded Spiking Fourier Transform in Neuromorphic Hardware".
Arxiv 2202.12650 (2022)_

For feedback, questions, or any comments on the library, you can contact us at lopez.randulfe@tum.de


# Installation

You can install the library as a python package using the following command:

    pip install git+https://github.com/KI-ASIC-TUM/time-coded-SFT.git

Alternatively, you can download the source and install the library locally.
This way, you can access and manipulate the provided examples:

    git clone https://github.com/KI-ASIC-TUM/time-coded-SFT.git
    pip install time-coded-SFT


# Usage

We provide a few examples for showing the basic operation of the model.

For getting a general understanding on how to simulate the SNN, we recommend going through the `sft_example.ipynb` notebook in the examples folder. We also provide a few python scripts that run different experiments with the S-FT:

* `experiment_layers_spikemap.py`, which generates a raster plot of the spikes at the different layers of the network, together with the membrane voltage evolution. It is useful for getting a visual inspection of the spike distribution over the time domain.
* `experiment_multiple_params.py`, which generates a plot of the S-FT performance for different simulation times and sample sizes
* `generate_snn_diagram.py`, which generates a sketch of the neuron model
* `run_sft.py`, which serves as the main experiment for testing the S-FT. It runs the SNN with the specified input data and network configuration. After the simulation, it adapts the output for generating a plot of the frequency spectrum, split into the real, imaginary, and absolut components of the FT. The plot compares the result of the S-FT with that of a traditional FFT 

For trying out different configurations, you can create your own _json_ config file and pass it to the `run_sft.py` script using the following syntax:


    python3 run_sft.py [-s] [-f] --config <path-to-your-config-file>

    -s Indicates if you want to show the resulting plot after generating it
    -f Indicates if you want to use output data from previous simulation stored
       in a file, instead of running again the SNN



In the _config_ folder you can find examples of configuration files. The parameters are the following:

* _sim_time:_ Simulation time for each neuron stage. Each layer contains two stages (silent and spiking)
* _time_step:_ Time elapsed per simulation step. It must be `1` for simulations in Loihi
* _mode:_ Choose between _dft_ or _fft_
* _framework:_ Choose between _numpy_, _brian_, or _loihi_
* _current_decay_: Decay of input current over time. For replicating the original S-FT, the value should be `0`
* _measure_performance:_ meant for measuring the energy and time performance in Loihi, this functinoality is not yet implemented
* _samples_per_chirp_: Number of samples taken. If the _fft_ mode is chosen, the number of samples N can only be a power of 4, i.e., 16, 64, 256, 1024
* _chirps_per_frame:_ How many chirps from the dataset to simulate
* _antennas:_  Number of radar antennas to simulate. With the available data, only `1` can be chosen
* _nframes:_  Number of radar frames to simulate. Right now, only `1` can be chosen


## Disclaimer

For simulating the S-FT in Loihi, the `NxSDK` library from intel needs to be installed. This library is not publicly available, and interested developers shall contact the Intel Neuromorphic Research Community (INRC) for getting access to it.