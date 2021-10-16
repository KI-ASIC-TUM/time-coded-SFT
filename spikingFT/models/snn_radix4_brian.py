
#!/usr/bin/env python3
"""
Module containing a class for implementing the S-FT in Loihi
"""
# Standard libraries
import logging
logger = logging.getLogger('spiking-FT')
import numpy as np
import pathlib
# Local libraries
try:
    import brian2 as brian
except ImportError:
    logger.warn("Brian2 cannot be found. "
                "It will not be possible to run simulation with Brian2.")
import spikingFT.models.snn_radix4

class SNNRadix4Brian(spikingFT.models.snn_radix4.FastFourierTransformSNN):
    """
    Class for setting up a network in Brian2 for the spiking FT

    Attributes:
    TODO: add attributes
    """


    PLATFORM = "brian"

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

        self.l_thresholds = []
        self.l_offsets = []
        if self.PLATFORM == 'loihi':
            axis = 1
        elif self.PLATFORM =='brian':
            axis = 0
        else:
            axis = 0

        for l in range(self.nlayers):
            self.l_weights[l] = 2*np.floor(self.l_weights[l]*127)

            #self.l_thresholds.append(8*254*self.sim_time/2)
            self.l_thresholds.append(2*np.floor(np.max(np.sum(np.abs(self.l_weights[l]),
                axis=axis))*(self.sim_time)/2/2))

            self.l_offsets.append(2*np.floor(np.sum(self.l_weights[l], axis =
                axis)*self.sim_time/2/2))

        # Initialize NETWORK
        self.layers = self.init_layers()
        self.synapses = []
        self.aux_neurons, self.aux_synapses = self.init_auxillary_neurons()
        self.net = None


        # PROBES
        self.l_probes_V = []
        self.l_probes_S = []
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
        
        layers = [] # nlayers list of compartment groups
        for l in range(self.nlayers):
            layer = brian.NeuronGroup(2*self.nsamples, self.neuron_model, 
                                threshold='v > ' + str(self.l_thresholds[l]), reset='v = 0; g = 0',
                                refractory=self.total_sim_time*brian.ms,
                                method='rk2', 
                                name='layer'+str(l)) 
            layer[:].v = -self.l_offsets[l]
            layers.append(layer)

        logger.debug('Done.')

        return layers

    def init_inputs(self, real_encoded_data, imag_encoded_data):
        """
        Initializes input layer of SNN with the given encoded data.

        Parameters:
            real_encoded_data (array): dimensions (nsamples/1) ttfs encoding
            imag_encoded_data (array): dimensions (nsamples/1) ttfs encoding
        
        Returns:
            layers: list of layers, each with 2*nsamples neurons
        """

        logger.debug('Initiliazing input layer and inputs ...')

        indices = np.arange(2*self.nsamples)
        encoded_data = np.hstack([real_encoded_data, imag_encoded_data])*brian.ms
        input_layer = brian.SpikeGeneratorGroup(2*self.nsamples, indices, encoded_data, name='input_layer')
        
        self.layers.append(input_layer)
        logger.debug('Done.')

        return

    def init_auxillary_neurons(self):
        """
        Initializes auxillary neurons such as clock neurons.

        Returns:
            clock_neurons: list of auxillary neurons
            clock_synapses: list of auxillary synapses
        """

        logger.debug('Creating auxillary neurons and synapses ...')

        clock_neurons = []
        clock_synapses = []
        for l in range(self.nlayers):
          spike_times = np.array(np.arange((l+1)*self.sim_time, self.total_sim_time, self.total_sim_time))*brian.ms
          indices = np.array([0]*len(spike_times))
          clock = brian.SpikeGeneratorGroup(1, indices, spike_times, name='clock_neurons' + str(l))
          clock_neurons.append(clock)

          clock_synapse = brian.Synapses(clock, self.layers[l], on_pre='g =' + str(2*self.l_thresholds[l]/self.sim_time))
          clock_synapse.connect()
          clock_synapses.append(clock_synapse)

        logger.debug('Auxillary neurons connected.')

        return clock_neurons, clock_synapses

    def init_probes(self):
        """
        Initializes probes of all compartment groups
        
        Returns:
            l_probes_V: list of voltage probes
            l_probes_S: list of spike probes
        """
        logger.debug('Creating Probes ...')

        #self.l_probes_S.append(SpikeMonitor(self.layers[-1]))

        for l in range(self.nlayers): 
          probes_S = (brian.SpikeMonitor(self.layers[l], name='probes_spikes'+str(l)))
          probes_V = (brian.StateMonitor(self.layers[l], 'v', record=np.arange(0,2*self.nsamples), 
                                    name='probes_voltage'+str(l)))
          self.l_probes_S.append(probes_S)
          self.l_probes_V.append(probes_V)

        logger.debug('Done.')

        return


    def connect(self):
        """
        Connect all layers (including input layer).

        """


        logger.debug('Connecting layers ...')

        for l in range(self.nlayers):

            # create synapses
            synapse = brian.Synapses(self.layers[l-1], self.layers[l], 
                                model='w : 1', on_pre='g += w', name='s'+str(l))

            # connect only non zero synapses
            sources, targets = self.l_weights[l].nonzero()
            synapse.connect(i=sources, j=targets)

            # set weights
            synapse.w = self.l_weights[l][sources,targets]
            self.synapses.append(synapse)

            logger.debug('Layer {0} connected to Layer {1}'.format(l, l+1))
        logger.debug('Done.')

        return

    def parse_probes(self):

        for l in range(self.nlayers):
            spike_times = [] 
            for t in self.l_probes_S[l].spike_trains():
                if not (self.l_probes_S[l].spike_trains()[t])/(1*brian.ms):
                    spike_times.append(0)
                else:
                    spike_times.append(self.l_probes_S[l].spike_trains()[t][0]/(1*brian.ms))

            self.spikes[:, 0, l] = spikingFT.utils.ft_utils.bit_reverse(np.array(spike_times)[:self.nsamples],
                                      base=4, nlayers=self.nlayers)
            self.spikes[:, 1, l] = spikingFT.utils.ft_utils.bit_reverse(np.array(spike_times)[self.nsamples:],
                                      base=4, nlayers=self.nlayers)

            self.voltage[:,:,0,l] = self.l_probes_V[l].v[:self.nsamples].T
            self.voltage[:,:,1,l] = self.l_probes_V[l].v[self.nsamples:].T
            
            self.output = (self.nlayers+0.5)*self.sim_time - self.spikes[:,:,-1]


    def simulate(self):

        logger.debug('Setting up network ... ')
        self.net = brian.Network(self.layers, self.synapses, 
                           self.aux_neurons, self.aux_synapses, 
                           self.l_probes_S, self.l_probes_V)
        logger.debug('Done.')

        logger.debug('Running Brian2 simulation ... ')
        self.net.run(self.total_sim_time * brian.ms)
        logger.info('Finishing Brian2 simulation ...')


    def run(self, data):

        # Create spike generators
        self.init_inputs(data.real, data.imag)
        # Connect all layers:W
        self.connect()
        # Init probes
        self.init_probes()


        # Run the network
        self.simulate()

        self.parse_probes()

        logger.debug('Simulation finished.')

        return self.output
