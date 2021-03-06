#!/usr/bin/env python3
"""
Module containing a class for implementing the S-FT in Loihi
"""
# Standard libraries
import logging
logger = logging.getLogger('spiking-FT')
import numpy as np
import pathlib
import os
# Local libraries
try:
    import nxsdk.api.n2a as nx
except ImportError:
    logger.warn("Intel NxSDK cannot be found. "
                "It will not be possible to run simulations with Loihi")
import spikingFT.models.snn
import spikingFT.utils.ft_utils

class SNNRadix4Loihi(spikingFT.models.snn.FourierTransformSNN):
    """
    Class for setting up a network on Loihi for the spiking FT

    Attributes:
        nsamples (int): determined by number of samples in radar data;
         determines number of layers and hence compartments in network
        nlayers (int): determined by number of samples (log4(nsamples))
        sim_time (int): charging phase time of network in loihi time steps;
         spiking phase also takes sim_time
        net: loihi network object
        current_decay (int): sets current decay of  alls synapses
        l_g: list compartment group with 2 * nsamples compartments for the
         coefficients (real, imaginary) of the DFT
        l_weights: list of connection weights for the compartments
        l_biases: list of biases for threshold offset
        board: loihi board which defines the compiler used
        bias_channel: channel for communicating bias data between ipynb and snip
        time_channel: channel for communicating time data between ipynb and snip
        clock_g: clock neurons using spike generation processes
        reset_g: reset neurons using spike generation processes
        l_probes_V: list of voltage probes of the compartments
        l_probes_S: list of spike probes of the compartments
        et_probe: execution time probe
        e_probe: energy consumption probe
    """
    PLATFORM = "loihi"
    FFT = True
    MAX_TH_MANT = 131071
    TH_MANT = (MAX_TH_MANT - int(MAX_TH_MANT/2) - 254)
    REFRACTORY_T = 63

    def __init__(self, **kwargs):
        """
        Initialize network
        
        Parameters:
            current_decay (int): sets current decay of synapses
            measure_performance (bool): Whether to perform a performance test
        """
        # Initialize parent class and unpack class-specific variables
        super().__init__(**kwargs)
        #os.environ['PARTITION'] = "nahuku32"
        #os.environ['BOARD'] = "ncl-ext-ghrd-02"
        self.current_decay = kwargs.get("current_decay")
        self.measure_performance = kwargs.get("measure_performance", False)
        self.probes_readout = 'last'

        # TODO: total sim time vs sim time
        self.total_sim_time = int(self.sim_time*(self.nlayers+2))

        # Initialize NETWORK PARAMETRS
        self.l_thresholds = []
        self.l_offsets = []
        self.weightExp = []
        axis = 1 if self.PLATFORM == 'loihi' else 0

        for l in range(self.nlayers):

            thresh = 4*254*self.sim_time/2
            weightExp = np.floor(np.log2(thresh/self.TH_MANT)+1)
            weightExp = 1 # best results so far
            self.weightExp.append(-weightExp)

            logger.debug('Weight rescaling:') 
            logger.debug('\tMaximum threshold: {0}'.format(self.TH_MANT))
            logger.debug('\tActual threshold: {0}'.format(thresh))
            logger.debug('\tlog2 scale: {0} '.format(np.log2(thresh/self.TH_MANT)))
            logger.debug('\tRescaling weights by 2**{0} = {1} '.format(-weightExp, 2**(-weightExp)))
            logger.debug('\tNew threshold: {0}'.format(4*254*self.sim_time/2*2**self.weightExp[l]))
            
            # factor of 0.8 is arbitrary and depends on data
            self.l_thresholds.append(0.8*4*254*self.sim_time/2*2**self.weightExp[l])
            #self.l_thresholds.append(np.max(np.sum(np.abs(weight_matrix), axis=axis))*(self.sim_time)/2)

            self.l_offsets.append(np.sum(self.l_weights[l]*2**self.weightExp[l], axis =
                axis)*self.sim_time/2)

        # Initialize NETWORK and COMPARTMENTS
        self.net = nx.NxNet()
        self.l_g = self.init_compartments()
        # init and connections
        self.clock_g, self.reset_g = self.init_auxillary_compartments() 

        # SNIP
        self.board = None
        self.bias_channel = None
        self.time_channel = None


        # PROBES
        self.l_probes_V, self.l_probes_S = self.init_probes()

        self.spikes = np.zeros((self.nsamples, 2, self.nlayers))
        self.voltage = np.zeros((self.total_sim_time, self.nsamples, 2, self.nlayers))
        
        self.et_probe = None
        self.e_probe = None
        return


    def init_compartments(self):
        """
        Initializes compartments depending of the number of samples
        
        Returns:
            l_g: list of compartment group with nsamples compartments for
             the coefficients of the DFT
        """

        logger.debug('Creating Compartments ...')
        core_distribution_factor = 128
        core_step = 64
        ncomp_per_layer = int(self.nsamples/16)
        
        l_g = [] # nlayers list of compartment groups
        for n in range(self.nlayers):
            l = []
            g = self.net.createCompartmentGroup()
            logger.debug('Creating CompartmentPrototypes of Layer {0} ...'.format(n))
            for i in range(self.nsamples*2):
                l_p = nx.CompartmentPrototype(
                    vThMant=self.l_thresholds[n] + self.l_offsets[n][i],
                    compartmentCurrentDecay=self.current_decay,
                    compartmentVoltageDecay=0,
                    functionalState=nx.COMPARTMENT_FUNCTIONAL_STATE.IDLE,
                    logicalCoreId=np.mod(i,core_step) +
                     np.mod(n,2)*int(self.nsamples*2/core_step),
                    #logicalCoreId=int(i/core_distribution_factor) +
                    # np.mod(n,2)*int(self.nsamples*2/core_distribution_factor),
                    #logicalCoreId=core_id[i],
                    refractoryDelay=self.REFRACTORY_T
                )
                l.append(self.net.createCompartment(prototype=l_p))
            g.addCompartments(l)
            l_g.append(g)

        logger.debug('Done.')

        return l_g

    def init_probes(self):
        """
        Initializes probes of all compartment groups
        
        Returns:
            l_probes_V: list of voltage probes
            l_probes_S: list of spike probes
        """
        logger.debug('Creating Probes ...')
        
        l_probes_V = []
        l_probes_S = []
        
        if self.probes_readout == 'all':
            for n in range(self.nlayers):
                logger.debug('Creating Probes of Layer {0} ...'.format(n))
                l_probes_V.append(self.l_g[n].probe(nx.ProbeParameter.COMPARTMENT_VOLTAGE,
                    probeConditions=None))
                l_probes_S.append(self.l_g[n].probe(nx.ProbeParameter.SPIKE,
                    probeConditions=nx.SpikeProbeCondition()))
        elif self.probes_readout == 'last':            
            logger.debug('Creating Probes of Layer {0} ...'.format(self.nlayers))
            customSpikeProbeCond = nx.SpikeProbeCondition(dt=1, tStart=(self.nlayers)*self.sim_time)
            l_probes_S.append(self.l_g[-1].probe(nx.ProbeParameter.SPIKE,
                probeConditions=customSpikeProbeCond))
        else:            
            logger.debug('No probes.')

    
        logger.debug('Done.')

        return l_probes_V, l_probes_S


    def init_inputs(self, real_encoded_data, imag_encoded_data):
        """
        Generates N real and N imaginary input generators, N=nsamples
        
        Parameters:
            real_encoded_data (array): dimensions (nsamples/1) ttfs encoding
            imag_encoded_data (array): dimensions (nsamples/1) ttfs encoding
        """
        logger.debug("Creating input spike generators")

        encoded_data = np.hstack([real_encoded_data, imag_encoded_data])

        gen = self.net.createSpikeGenProcess(numPorts=self.nsamples*2)
        
        # Specify spiking time of each generator
        logger.debug("Assigning spike times to input spike generators")
        for i in range(self.nsamples*2):
            gen.addSpikes(spikeInputPortNodeIds=i,
                               spikeTimes=[encoded_data[i]])

        self.l_g.append(gen)
        return



    def connect(self):
        """
        Connect all layers (including input layer)

        """

        #TODO: Splitting input and layer connection (?)

        logger.debug('Connecting layers ...')
        
        logger.debug('Creating connection prototype ...')
        
        # TODO: make weight tests
        
        #con2 = nx.ConnectionPrototype(signMode=2, weightExponent=self.weight_exp, compressionMode=3)
        #con3 = nx.ConnectionPrototype(signMode=3, weightExponent=self.weight_exp, compressionMode=3)
        
        inhibit_con = nx.ConnectionPrototype(signMode=3, weightExponent=4, compressionMode=0)

        logger.debug('Start connecting ...')
        for i in range(self.nlayers):
            weights2 = self.l_weights[i].copy()
            weights3 = self.l_weights[i].copy()
            idx2 = self.l_weights[i] > 0
            idx3 = self.l_weights[i] < 0
            weights2[idx3] = 0
            weights3[idx2] = 0

            con = nx.ConnectionPrototype(signMode=1, weightExponent=self.weightExp[i],
                compressionMode=0)
            
            #self.l_g[i-1].connect(self.l_g[i], prototype=con2,
            #        weight=weights2)
            #self.l_g[i-1].connect(self.l_g[i], prototype=con3,
            #        weight=weights3)
            mask = np.abs(self.l_weights[i])>1
            #if i==0:
            #    mask[self.nsamples:,self.nsamples:] = 0
            #elif i==self.nlayers-1:
            #    mask[self.nsamples:,:] = 0
            logger.debug('Number of connections in layer {0}: {1} of {2}'.format(i, np.sum(mask), self.nsamples**2*4))

            self.l_g[i-1].connect(self.l_g[i], prototype=con,
                    weight=self.l_weights[i], connectionMask=mask)
            self.l_g[i].connect(self.l_g[i], prototype=inhibit_con,
                    weight=-254*np.eye(2*self.nsamples),
                    connectionMask=np.eye(2*self.nsamples))
            logger.debug('Layer {0} connected to Layer {1}'.format(i, i+1))
        logger.debug('Done.')

        return

    def init_snip(self):
        """
        Initialize SNIPs: Set up of board, compile and snip

        There are two communication channels available: one for the bias
        and one for the time
        """
        logger.debug("Compiling n2board")
        compiler = nx.N2Compiler()
        self.board = compiler.compile(self.net)
        root_path = pathlib.Path(__file__).resolve().parent.parent
        snip_path = root_path.joinpath("snips/bias_snip")
        logger.debug("Importing snips from {}".format(snip_path))
        mgmt_snip = self.board.createProcess(
            name="runMgmt",
            includeDir=snip_path.as_posix(),
            cFilePath=snip_path.joinpath("setBias.c").as_posix(),
            funcName="runMgmt",
            guardName="doRunMgmt",
            phase="mgmt"
        )

        self.bias_channel = self.board.createChannel(b'biasChannel', "int", 1)
        self.time_channel = self.board.createChannel(b'timeChannel', "int", 1)

        self.bias_channel.connect(None, mgmt_snip)
        self.time_channel.connect(mgmt_snip, None)

        logger.debug("Starting n2board driver")
        self.board.start()

    def init_auxillary_compartments(self):
        """
        Initialize auxiallary neurons and connect them to the network.
        Auxillary neuron serve as clock neuron and reset neuron.

        """

        logger.debug('Creating ConnectionPrototype ...')
        clock_g = []
        reset_g = []
        reset_con = nx.ConnectionPrototype(signMode=3, weightExponent=4, compressionMode=0)
        
        logger.debug('Creating auxillary neurons ...')
        
        
        for i in range(self.nlayers):
            

            slope = 2*self.l_thresholds[i]/self.sim_time*np.ones(2*self.nsamples)-np.sum(self.l_weights[i], axis=1)*2**self.weightExp[i]
            scale = slope.max()/254
            weightExp = np.floor(np.log2(slope.max()/254)+1-1e-8)
            
            logger.debug('Weight rescaling:') 
            logger.debug('\tMaximum weight: {0}'.format(254))
            logger.debug('\tActual weight: {0}'.format(slope.max()))
            logger.debug('\tlog2 scale: {0} '.format(np.log2(scale)))
            logger.debug('\tRescaling by 2**{0} = {1} '.format(weightExp, 2**(-weightExp)))
            logger.debug('\tNew weight: {0}'.format(slope.max()*2**(-weightExp)))

            clock_con = nx.ConnectionPrototype(signMode=1, weightExponent=weightExp, compressionMode=0)
            
            clock = self.net.createSpikeGenProcess(numPorts=1)
            clock.addSpikes(spikeInputPortNodeIds=0, spikeTimes=[self.sim_time*(i+1)])
            clock.connect(self.l_g[i], prototype=clock_con,
                    weight=slope/2**weightExp)
            clock_g.append(clock)
            logger.debug('Clock neuron connected to layer {0}.'.format(i))
            
            #reset = self.net.createSpikeGenProcess(numPorts=1)
            #reset.addSpikes(spikeInputPortNodeIds=0,
            #        spikeTimes=[(i+2)*self.sim_time+4])
            #reset.connect(self.l_g[i], prototype=reset_con,
            #        weight=-254*np.ones(self.nsamples*2))
            #reset_g.append(reset)
            
            logger.debug('Reset neuron connected to layer {0}.'.format(i))
        logger.debug('Done.')

        return clock_g, reset_g



    def performance_profiling(self):
        """
        Perform a performance profiling of the SNN in loihi

        Instantiate probes to measure both energy and execution time;
        energy is measured by the host CPU while the execution time is
        measured by one of the embedded Lakemont (LMT) CPUs

        time stamps t_H and t_LMT of the two probes differ
        """
        logger.debug('Compiling ...')

        compiler = nx.N2Compiler()
        self.board = compiler.compile(self.net)
        logger.debug('Done.')

        

        logger.debug("Instantiating performance boards")
        self.et_probe = self.board.probe(
            probeType=nx.ProbeParameter.EXECUTION_TIME,
            probeCondition=nx.PerformanceProbeCondition(
                tStart=1,
                tEnd=self.total_sim_time,
                bufferSize=1024,
                binSize=2)
        )

        self.e_probe = self.board.probe(
            probeType=nx.ProbeParameter.ENERGY,
            probeCondition=nx.PerformanceProbeCondition(
                tStart=1,
                tEnd=self.total_sim_time,
                bufferSize=1024,
                binSize=2)
        )


    def simulate_with_snip(self):

        #TODO: Use snip instead of auxiallary neurons

        charging_stage_bias = int(self.l_thresholds[0]*2/self.sim_time)
        logger.debug("Running simulation")
        # Write bias value during charging stage to the corresponding channel
        self.bias_channel.write(1, [charging_stage_bias])
        # Run charging stage
        self.board.run(self.sim_time*2, aSync=True)
        # Run spiking stage
        spiking_stage_bias = np.zeros(2, int)
        for bias in spiking_stage_bias[1:]:
            self.time_channel.read(1)[0]
            self.bias_channel.write(1, [bias])
        # Finish and disconnect
        logger.info("Finishing Loihi execution. Disconnecting board")
        # Run spiking stage
        #self.board.finishRun()
        self.board.disconnect()

    def simulate(self):

        logger.debug('Compiling ...')

        compiler = nx.N2Compiler()
        self.board = compiler.compile(self.net)
        logger.debug('Done.')
        
        logger.debug('start board ...')
        self.board.start()
        logger.debug('done.')

        logger.debug('Running simulation ... ')
        self.board.run(self.total_sim_time, aSync=True)

        logger.info('Finishing Loihi execution. Disconnecting board ...')
        self.board.finishRun()
        self.board.disconnect()
        #self.power_stats = self.board.energyTimeMonitor.powerProfileStats

    def parse_probes(self):
        if self.probes_readout == 'all':
            for l in range(self.nlayers):
                self.voltage[:, :, 0, l] = self.l_probes_V[l][0].data[:self.nsamples,:].T
                self.voltage[:, :, 1, l] = self.l_probes_V[l][0].data[self.nsamples:,:].T

                real_spikes = np.argmax(self.l_probes_S[l][0].data[:self.nsamples], axis=1)
                imag_spikes = np.argmax(self.l_probes_S[l][0].data[self.nsamples:], axis=1)
                self.spikes[:, 0, l] = spikingFT.utils.ft_utils.bit_reverse(real_spikes,
                        base=4, nlayers=self.nlayers)
                self.spikes[:, 1, l] = spikingFT.utils.ft_utils.bit_reverse(imag_spikes,
                        base=4, nlayers=self.nlayers)
        elif self.probes_readout == 'last':
            real_spikes = np.argmax(self.l_probes_S[-1][0].data[:self.nsamples], axis=1)
            imag_spikes = np.argmax(self.l_probes_S[-1][0].data[self.nsamples:], axis=1)
            self.spikes[:, 0, -1] = spikingFT.utils.ft_utils.bit_reverse(real_spikes,
                    base=4, nlayers=self.nlayers)
            self.spikes[:, 1, -1] = spikingFT.utils.ft_utils.bit_reverse(imag_spikes,
                    base=4, nlayers=self.nlayers)
        else:
            logger.debug('No probes.')

        logger.debug(self.spikes[:,:,-1])
        self.output = (0.5)*self.sim_time - self.spikes[:,:, -1] + 1.5
        logger.debug(self.output)


    def parse_energy_probe(self):
        """
        Calculate consumed energy from the collected power stats
        """
        power_vals = np.array(self.power_stats.power['core']['dynamic'])
        time_vals = np.array(self.power_stats.timePerTimestep)
        energy = power_vals * time_vals
        return energy



    def run(self, data):

        # Create spike generators
        self.init_inputs(data.real, data.imag)
        # Connect all layers
        self.connect()

        # Instantiate measurement probes
        #if self.measure_performance:
        #    self.performance_profiling()

        #TODO: use snip
        #self.init_snip()


        # Run the network
        #
        self.simulate()
        logger.debug('Done.')
         
        #if self.measure_performance:
        #    self.parse_energy_probe()

        self.parse_probes()
        logger.debug('Run finished.')
        return self.output
