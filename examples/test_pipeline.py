#!/usr/bin/env python3
"""
Module one-line definition
"""
# Standard libraries
import matplotlib.pyplot as plt
import numpy as np
import pathlib
# Local libraries
import spikingFT.startup
import spikingFT.sim_handler


if __name__ == "__main__":
    # Simulate network with configuration stored in local file
    conf_file = "../config/test_experiment.json"
    sim = spikingFT.startup.startup(conf_file)
    sim_time = sim.snn.sim_time
    nsamples = sim.snn.nsamples

    # Plot output from S-FT
    sr = []
    si = []
    for i in range(1, nsamples):
        spike_data = np.array(sim.snn.l1_real_probes_S[0][i].data)
        spike_t = np.where(spike_data==1)
        single_spike_t = spike_t[0][0] if spike_t[0].size != 0 else sim_time*1.5
        sr.append(single_spike_t-sim_time*1.5)

    for i in range(1, nsamples):
        spike_data = np.array(sim.snn.l1_imag_probes_S[0][i].data)
        spike_t = np.where(spike_data==1)
        single_spike_t = spike_t[0][0] if spike_t[0].size != 0 else sim_time*1.5
        si.append(single_spike_t-sim_time*1.5)

    sdft = np.sqrt(np.array(sr)**2 + np.array(si)**2)
    result = np.split(sdft ,[int(nsamples/2), int(nsamples/2)])

    # Plot standard 1D FFT
    chirp_data = sim.encoded_data[0, :]
    fft = np.abs(np.fft.fft(chirp_data))
    ax1 = plt.subplot(2, 1, 1)
    plt.title('Numpy FFT Result')
    ax1.plot(fft[1:int(fft.size/2)])

    ax2 = plt.subplot(2, 1, 2)
    plt.title('SDFT Result')
    ax2.plot(result[0])

    plt.tight_layout()
    plt.show()
