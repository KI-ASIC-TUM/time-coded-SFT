#!/usr/bin/env python3
"""
Module one-line definition
"""
# Standard libraries
import logging
import numpy as np
import time
# Local libraries
import spikingFT.models.snn

logger = logging.getLogger('spiking-FT')


class SNNNumpy(spikingFT.models.snn.FourierTransformSNN):
    PLATFORM = "numpy"

    def __init__(self, **kwargs):
        """
        Class for setting up a network on NumPy for the spiking FT
        """
        super().__init__(**kwargs)

        # SNN simulation parameters
        self.current_time = 0
        self.sim_time = kwargs.get("sim_time")
        # Neuron properties
        self.v_threshold = 1
        # Network variables
        self.n_chirps = 1
        self.n_input =  kwargs.get("nsamples")
        self.spikes = np.zeros((self.n_input, 2*self.n_chirps))
        self.voltage = np.zeros((int(2*self.sim_time), self.n_input, 2*self.n_chirps))
        self.l1 = self.init_compartments()

    def init_compartments(self):
        """
        Initializes compartments depending on the number of samples
        """
        l1 = SpikingNeuralLayer(
                                (self.n_input, 2*self.n_chirps),
                                (self.real_weights, self.imag_weights),
                                v_threshold=self.v_threshold,
                                time_step=1
                               )
        return l1

    def simulate(self, spike_trains):
        logger.info("Layer 1: Charging stage")
        # Charging stage of layer 1
        while self.current_time <= self.sim_time:
            causal_neurons_l1 = (spike_trains < self.sim_time).reshape(
                    (self.n_chirps, self.n_input)
                )
            out_l1 = self.l1.update_state(causal_neurons_l1) * self.current_time
            self.spikes += out_l1
            self.voltage[int(self.current_time)] = self.l1.v_membrane
            self.current_time += 1

        logger.info("Layer 1: Spiking stage")
        # Spiking stage of layer 1, and charging stage of layer 2
        self.l1.bias = (2 * self.v_threshold) / self.sim_time
        causal_neurons_l1 = np.zeros_like(causal_neurons_l1)
        while self.current_time < 2*self.sim_time:
            out_l1 = self.l1.update_state(causal_neurons_l1)
            self.spikes +=  out_l1 * (self.current_time)
            self.voltage[int(self.current_time)] = self.l1.v_membrane
            self.current_time += 1
        # All neurons that didn't spike are forced to spike in the last step,
        # since the spike-time of 1 corresponds to the lowest possible value.
        self.spikes = np.where(self.spikes == 0, 1.5*self.sim_time, self.spikes)
        self.spikes -= 1.5*self.sim_time

    def run(self, spike_trains):
        """
        Main routine for running the simulation
        """
        spike_trains = spike_trains.real
        logging.info("Running the NumPy TTFS Spiking-DFT")

        self.simulate(spike_trains)
        return self.spikes


class SpikingNeuralLayer():
    """
    Class for implementing a single spiking-DFT neural layer

    Args:
        shape (int|list): number of neurons in the layer. Int for a 1D
            layer or an iterable of ints for N-D layers
        weights (np.array): Matrix containing the input weights to the
            layer. They have to be real numbers
        **bias (double): external current fed to the neurons
        **threshold (double): membrane voltage for generating a spike
        **time_step (double): time gap between iterations
    """
    def __init__(self, shape, weights, **kwargs):
        """
        Initialize the class
        """
        # Neuron properties
        self.bias = kwargs.get("bias", 0)
        self.v_threshold = kwargs.get("v_threshold", 0.05)
        # Neuron variables
        self.v_membrane = np.zeros(shape)
        self.spikes = np.zeros(shape)
        self.refactory = np.zeros(shape)
        self.weights = weights

        # Simulation parameters
        self.time_step = kwargs.get("time_step", 0.001)

    def update_input_currents(self, input_spikes):
        """
        Calculate the total current that circulates inside each neuron
        """
        # Calculate separately the currents to the real and imaginary neurons
        z_real = np.dot(self.weights[0], input_spikes.transpose())
        z_imag = np.dot(self.weights[1], input_spikes.transpose())
        z = np.hstack((z_real, z_imag))
        # Add bias to the result and multiply by time_step
        z += self.bias
        z *= self.time_step
        return z

    def update_membrane_potential(self, z):
        """
        Update membrane potential of each neuron

        The membrane potential increases based on the input current, and
        it returns to the rest voltage after a spike
        """
        self.v_membrane += z
        self.v_membrane *= (1-self.refactory)
        return self.v_membrane

    def generate_spikes(self):
        """
        Determine which neurons spike, based on membrane potential
        """
        # Generate a spike when the voltage is higher than the threshold
        self.spikes = np.where((self.v_membrane>self.v_threshold), True, False)
        # Activate the refactory period for the neurons that spike
        self.refactory += self.spikes
        return self.spikes

    def update_state(self, in_spikes):
        """
        Update internal state of neurons, based on input spikes
        """
        z = self.update_input_currents(in_spikes)
        self.update_membrane_potential(z)
        out_spikes = self.generate_spikes()
        return out_spikes
