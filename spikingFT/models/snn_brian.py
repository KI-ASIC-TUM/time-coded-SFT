#!/usr/bin/env python3
"""
Module containing a class for implementing the S-FT in Brian2
"""
# Standard libraries
import logging
logger = logging.getLogger('spiking-FT')
import spikingFT.models.snn
import numpy as np
import matplotlib.pyplot as plt

# Local libraries
try:
    import brian2 as brian
except ImportError:
    logger.warn("Brian2 cannot be found. "
                "It will not be possible to run simulation with Brian2.")

class SNNBrian(spikingFT.models.snn.FourierTransformSNN):
    """
    Class for setting up a network in Brian2 for the spiking FT

    Attributes:
    time_step (float): size of time step 
    total_sim_time (float): (n_layer+1)*sim_time
    l_threshold (list): thresholds per layer
    l_offsets (list): offsets per layer
    l_weights (list): weight matrices per layer
    layers (list): list of brian2 population
    synapses (list): list of brian2 synapses
    aux_neurons (list): list of brian2 spike generators
    aux_synapses (list): list of brian2 synapses
    net (object): brian2 network
    l_probes_S (list): list of brian2 spike probes
    l_probes_V (list): list of brian2 state probes
    spikes (array): array containing spikes
    voltages (array): array containing voltage values
    """


    PLATFORM = "brian"
    FFT = False

    def __init__(self, **kwargs):
        """
        Initialize network
        
        Parameters:
        TODO: add parameters
        """
        # Initialize parent class and unpack class-specific variables
        super().__init__(**kwargs)

        # Initialize SIMULATION parameters
        self.time_step = kwargs.get("time_step")
        self.total_sim_time = (self.nlayers+1)*self.sim_time

        # Initialize NEURON MODEL
        self.neuron_model = self.init_neuron_model()

        # Initialize NETWORK PARAMETRS

        self.v_threshold =  np.sum(self.real_weights[0,:]) * self.total_sim_time / 2
        print(self.v_threshold)

        # Initialize NETWORK
        self.re_input = None
        self.im_input = None
        self.re_re_synapse = None
        self.im_re_synapse = None
        self.re_im_synapse = None
        self.im_im_synapse = None
        self.re_charging_neurons, self.im_charging_neurons = self.init_layers()
        self.aux_neurons, self.re_aux_synapses, self.im_aux_synapses = self.init_auxillary_neurons()
        self.net = None


        # PROBES
        self.re_probes_S, self.im_probes_S, self.re_probes_V, self.im_probes_V = self.init_probes()
        self.spikes = np.zeros((self.nsamples, 2, self.nlayers))
        self.voltage = np.zeros((int(self.total_sim_time/self.time_step), self.nsamples, 2, self.nlayers))
        
        return

    def init_neuron_model(self):
        """
        Initializes equation of neuron model.
        
        Returns:
            eq (String): equation of neuron model
        """

        eq = '''
        dv/dt = g/(1*ms): 1 (unless refractory)
        g : 1
        '''

        return eq



    def init_layers(self):
        """
        Initializes layer of SNN with the given neuron model.
        
        Returns:
            layers: list of layers, each with 2*nsamples neurons
        """

        logger.debug('Creating layers ...')
        
        sin_charging_neurons = brian.NeuronGroup(self.nsamples, self.neuron_model, threshold='v > ' + str(self.v_threshold), reset='v = 0; g = 0',
                                refractory=0*brian.ms, method='euler', name='im_charging_neurons')
        sin_charging_neurons[0].g = 0
        cos_charging_neurons = brian.NeuronGroup(self.nsamples, self.neuron_model, threshold='v > ' + str(self.v_threshold), reset='v = 0; g = 0',
                                refractory=0*brian.ms, method='euler', name='re_charging_neurons')
        cos_charging_neurons[0].g = 0

        logger.debug('Done.')

        return cos_charging_neurons, sin_charging_neurons


    def init_inputs(self, encoded_data):
        """
        Initializes input layer of SNN with the given encoded data.

        Parameters:
            real_encoded_data (array): dimensions (nsamples/1) ttfs encoding
            imag_encoded_data (array): dimensions (nsamples/1) ttfs encoding
        
        Returns:
            layers: list of layers, each with 2*nsamples neurons
        """

        logger.debug('Initiliazing input layer and inputs ...')

        indices = np.arange(self.nsamples)
        re_input_layer = brian.SpikeGeneratorGroup(self.nsamples, indices, encoded_data.real * brian.ms, name='re_input_neurons')
        im_input_layer = brian.SpikeGeneratorGroup(self.nsamples, indices, encoded_data.imag * brian.ms, name='im_input_neurons')
        
        logger.debug('Done.')

        return re_input_layer, im_input_layer

    def init_auxillary_neurons(self):
        """
        Initializes auxillary neurons such as clock neurons.

        Returns:
            clock_neurons: list of auxillary neurons
            clock_synapses: list of auxillary synapses
        """

        logger.debug('Creating auxillary neurons and synapses ...')

        spike_times = np.array(np.arange(self.sim_time, self.total_sim_time, self.total_sim_time))*brian.ms
        indices = np.array([0]*len(spike_times))
        clock_neuron = brian.SpikeGeneratorGroup(1, indices, spike_times, name='clock_neuron')

        re_clock_synapse = brian.Synapses(clock_neuron, self.re_charging_neurons, on_pre='g =' + str(2*self.v_threshold/self.sim_time))
        re_clock_synapse.connect()
        im_clock_synapse = brian.Synapses(clock_neuron, self.im_charging_neurons, on_pre='g =' + str(2*self.v_threshold/self.sim_time))
        im_clock_synapse.connect()

        logger.debug('Auxillary neurons connected.')

        return clock_neuron, re_clock_synapse, im_clock_synapse

    def init_probes(self):
        """
        Initializes probes of all compartment groups
        
        Returns:
            l_probes_V: list of voltage probes
            l_probes_S: list of spike probes
        """
        logger.debug('Creating Probes ...')

        re_probes_S = (brian.SpikeMonitor(self.re_charging_neurons, name='re_probes_spikes'))
        re_probes_V = (brian.StateMonitor(self.re_charging_neurons, 'v', record=np.arange(0,self.nsamples), 
                                  name='re_probes_voltage'))

        im_probes_S = (brian.SpikeMonitor(self.im_charging_neurons, name='im_probes_spikes'))
        im_probes_V = (brian.StateMonitor(self.im_charging_neurons, 'v', record=np.arange(0,self.nsamples), 
                                  name='im_probes_voltage'))

        logger.debug('Done.')

        return re_probes_S, im_probes_S, re_probes_V, im_probes_V


    def connect(self):
        """
        Connect all layers (including input layer).

        """


        logger.debug('Connecting layers ...')

        re_re_synapse = brian.Synapses(self.re_input, self.re_charging_neurons, 
                                        model='w : 1', on_pre='g += w', name='re_re_synapses') 
        sources, targets = self.real_weights.nonzero()
        re_re_synapse.connect(i=sources, j=targets)
        re_re_synapse.w = self.real_weights[sources, targets]

        im_re_synapse = brian.Synapses(self.im_input, self.re_charging_neurons, 
                                        model='w : 1', on_pre='g += w', name='im_re_synapses') 
        sources, targets = self.imag_weights.nonzero()
        im_re_synapse.connect(i=sources, j=targets)
        im_re_synapse.w = self.imag_weights[sources, targets]

        re_im_synapse = brian.Synapses(self.re_input, self.im_charging_neurons, 
                                        model='w : 1', on_pre='g += w', name='re_im_synapses') 
        sources, targets = (+self.imag_weights).nonzero()
        re_im_synapse.connect(i=sources, j=targets)
        re_im_synapse.w = (+self.imag_weights)[sources, targets]

        im_im_synapse = brian.Synapses(self.im_input, self.im_charging_neurons, 
                                        model='w : 1', on_pre='g += w', name='im_im_synapses') 
        sources, targets = (-self.real_weights).nonzero()
        im_im_synapse.connect(i=sources, j=targets)
        im_im_synapse.w = (-self.real_weights)[sources, targets]

        logger.debug('Done.')

        return re_re_synapse, im_re_synapse, re_im_synapse, im_im_synapse

    def parse_probes(self):

        re_spike_times = [] 
        for t in self.re_probes_S.spike_trains():
            if not len((self.re_probes_S.spike_trains()[t])/(1*brian.ms)) > 0:
                re_spike_times.append(0)
            else:
                re_spike_times.append(self.re_probes_S.spike_trains()[t][0]/(1*brian.ms))
        
        im_spike_times = [] 
        for t in self.im_probes_S.spike_trains():
            if not len((self.im_probes_S.spike_trains()[t])/(1*brian.ms)) > 0:
                im_spike_times.append(0)
            else:
                im_spike_times.append(self.im_probes_S.spike_trains()[t][0]/(1*brian.ms))

        self.spikes[:, 0, 0] = np.array(re_spike_times)
        self.spikes[:, 1, 0] = np.array(im_spike_times)

        self.voltage[:,:,0, 0] = self.re_probes_V.v.T
        self.voltage[:,:,1, 0] = self.im_probes_V.v.T
        
        self.output = (self.nlayers+0.5)*self.sim_time - self.spikes[:,:,-1]


    def simulate(self):

        logger.debug('Setting up network ... ')
        self.net = brian.Network(self.re_charging_neurons, self.im_charging_neurons,
                                self.re_input, self.im_input, 
                                self.re_re_synapse, self.im_re_synapse, self.re_im_synapse, self.im_im_synapse,
                                self.aux_neurons, self.re_aux_synapses, self.im_aux_synapses, 
                                self.re_probes_S, self.re_probes_V,
                                self.im_probes_S, self.im_probes_V)
        logger.debug('Done.')

        logger.debug('Running Brian2 simulation ... ')
        self.net.run(self.total_sim_time * brian.ms)
        logger.info('Finishing Brian2 simulation ...')


    def run(self, data):

        # Create spike generators
        self.re_input, self.im_input = self.init_inputs(data)
        # Connect all layers:W
        self.re_re_synapse, self.im_re_synapse, self.re_im_synapse, self.im_im_synapse = self.connect()
        # Init probes
        self.init_probes()


        # Run the network
        self.simulate()

        self.parse_probes()

        logger.debug('Simulation finished.')

        return self.output
