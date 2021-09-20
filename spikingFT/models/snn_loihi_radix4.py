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
    import nxsdk.api.n2a as nx
except ImportError:
    logger.warn("Intel NxSDK cannot be found. "
                "It will not be possible to run simulations with Loihi")
import spikingFT.models.snn

# TODO: import

def twiddle_factor(k,N, type='exp'):
  '''
  Return twiddle factors.
  '''
  if type=='cos':
    return np.cos(2*np.pi*k/N)
  elif type=='sin':
    return np.sin(2*np.pi*k/N)
  elif type=='exp':
    return np.exp(2j*np.pi*k/N)

def loihi_normalization(weight_mask):
    weight_mask_norm = np.ceil(127 * weight_mask - 0.5)
    return weight_mask_norm*2



class SNNLoihi(spikingFT.models.snn.FourierTransformSNN):
    """
    Class for setting up a network on Loihi for the spiking FT

    Attributes:
        nsamples (int): determined by number of samples in radar data;
         determines number of layers and hence compartments in network
        nlayers (int): determined by number of samples (log4(nsamples))
        Tc (int): charging/spiking cycle time
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
        l_probes_V: list of voltage probes of the compartments
        l_probes_S: list of spike probes of the compartments
        et_probe: execution time probe
        e_probe: energy consumption probe
    """
    PLATFORM = "loihi"
    MAX_TH_MANT = 131071
    TH_MANT = int(MAX_TH_MANT/2)
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
        self.current_decay = kwargs.get("current_decay")
        self.measure_performance = kwargs.get("measure_performance", False)

        # WEIGHTS and BIASES
        self.l_weights = []
        self.l_biases = []
        for n in range(nlayers):
            weight_matrix = loihi_normalization(self.init_weight_matrix(n))
            self.l_weights.append(weight_matrix)
            self.biases.append(np.sum(weight_matrix, axis=1)*self.Tc/2)

        # Initialize NETWORK and COMPARTMENTS
        self.net = nx.NxNet()
        self.l_g = self.init_compartments()

        # SNIP
        self.board = None
        self.bias_channel = None
        self.time_channel = None

        # PROBES
        self.l_probes_V, self.l_probes_S = self.init_probes()

        
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
        core_distribution_factor = int(self.nsamples/64)
        
        l_g = [] # nlayers list of compartment groups
        l = []
        g = self.net.createCompartmentGroup()
        for n in range(nlayers):
            logger.debug('Creating CompartmentPrototypes of Layer {0} ...'.format(n))
            for i in range(self.nsamples*2):
                l_p = nx.CompartmentPrototype(
                    vThMant=self.TH_MANT + self.l_biases[n][i],
                    compartmentCurrentDecay=self.current_decay,
                    compartmentVoltageDecay=0,
                    functionalState=nx.COMPARTMENT_FUNCTIONAL_STATE.IDLE,
                    logicalCoreId=int(i/core_distribution_factor) +
                     n*int(nsamples*2/core_distribution_factor),
                    refractoryDelay=self.REFRACTORY_T
                )
                g.append(self.net.createCompartment(prototype=l_p))
            l.addCompartments(g)
            l_g.append(l)

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

        for n in range(nlayers):
            logger.debug('Creating Probes of Layer {0} ...'.format(n))
            l_probes_V.append(self.l_g[n].probe(nx.ProbeParameter.COMPARTMENT_VOLTAGE,
                probeConditions=None))
            l_probes_S.append(self.l_g[n].probe(nx.ProbeParameter.SPIKE,
                probeConditions=None))

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

        gen = self.net.createSpikeGenProcess(numPorts=self.nsamples)
        
        # Specify spiking time of each generator
        logger.debug("Assigning spike times to input spike generators")
        for i in range(self.nsamples*2):
            gen.addSpikes(spikeInputPortNodeIds=i,
                               spikeTimes=[encoded_data[i]])

        self.l_g.append(gen)
        return

    def init_weight_matrix(self, l):
        """
        Initialize radix4 weight matrices
        """
        logger.debug("Creating radix4 weight matrix of layer {0}".format(l))

        radix = 4
        n_layers = int(np.log(nsamples)/np.log(radix)) # number of layers
        n_bfs = int(radix**n_layers/radix) # number of butterflies
        n_blocks = radix**(l) # number of blocks
        n_bfs_per_block = int(n_bfs/n_blocks) # number of butterflies in one block
        distance_between_datapoints = radix**(n_layers-l-1) 
        distance_between_blocks = radix**(l) 

        n = np.tile(np.arange(0,radix**(n_layers-l-1)),radix**(l+1))
        c = np.tile(np.repeat(np.arange(0,4**(l+1),4**l),4**(n_layers-l-1)),4**l)

          # radix4 butterfly
        W = np.array([[   1,   1,   1,   1],
                    [   1, -1j,  -1,  1j],
                    [   1,  -1,   1,  -1],
                    [   1,  1j,  -1,  -1j]])

          # build 4 matrices of radix4 butterfly
        W_rr = np.real(W) # real input real weights real ouput
        W_ir = -1*np.imag(W) # imag input imag weights real output
        W_ri = np.imag(W) # real input imag weights imag ouput
        W_ii = np.real(W) # imag input real weights imag output 

        # Fancy matrix structure 
        G = np.eye(4**l, 4**l)
        B = np.eye(int(nsamples/4**(l+1)), int(nsamples/4**(l+1)))

        # Kronecker magic
        M_rr = np.kron(G,np.kron(W_rr, B))
        M_ir = np.kron(G,np.kron(W_ir, B))
        M_ri = np.kron(G,np.kron(W_ri, B))
        M_ii = np.kron(G,np.kron(W_ii, B))

        # Stacking 4 matrices to 2x2 
        M = np.vstack([np.hstack([M_rr, M_ir]), np.hstack([M_ri, M_ii])])

        # twiddle factor as 2x2 matrix
        tf_real = np.hstack([np.diag(twiddle_factor(n*c,nsamples,'cos')), -np.diag(twiddle_factor(n*c,nsamples,'sin'))])
        tf_imag = np.hstack([np.diag(twiddle_factor(n*c,nsamples,'sin')), np.diag(twiddle_factor(n*c,nsamples,'cos'))])
        tf = np.vstack([tf_real, tf_imag])

        # Twiddle factors times butterfly matrix
        M = np.matmul(tf.T,M)
        return M


    def connect(self):
        """
        Connect all layers (including input layer)

        TODO: Splitting input and layer connection (?)
        """

        logger.debug('Connecting layers ...')
        
        logger.debug('Creating connection prototype ...')
        con = nx.ConnectionPrototype(signMode=1, weightExponent=0, compressionMode=0)

        print('Start connecting ...')
        for i in range(self.nlayers):
            self.l_g[i-1].connect(self.l_g[i], prototype=con,
                    weight=self.l_weights[i])
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
        self.board.startDriver()


    def performance_profiling(self):
        """
        Perform a performance profiling of the SNN in loihi

        Instantiate probes to measure both energy and execution time;
        energy is measured by the host CPU while the execution time is
        measured by one of the embedded Lakemont (LMT) CPUs

        time stamps t_H and t_LMT of the two probes differ
        """
        logger.debug("Instantiating performance boards")
        self.et_probe = self.board.probe(
            probeType=nx.ProbeParameter.EXECUTION_TIME,
            probeCondition=nx.PerformanceProbeCondition(
                tStart=1,
                tEnd=self.sim_time*2,
                bufferSize=1024,
                binSize=2)
        )

        self.e_probe = self.board.probe(
            probeType=nx.ProbeParameter.ENERGY,
            probeCondition=nx.PerformanceProbeCondition(
                tStart=1,
                tEnd=self.sim_time*2,
                bufferSize=1024,
                binSize=2)
        )

    def simulate(self):
        charging_stage_bias = int(self.TH_MANT*2/self.sim_time)
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
        self.board.finishRun()
        self.board.disconnect()


    def run(self, data):
        # Create spike generators and connect them to compartments
        self.connect_inputs(data.real, data.real)
        # Instantiate measurement probes
        if self.measure_performance:
            network.performance_profiling()
        self.init_snip()
        # Run the network
        self.simulate()

        voltage_probes = (self.l1_real_probes_V[0], self.l1_imag_probes_V[0])
        spike_probes = (self.l1_real_probes_S[0], self.l1_imag_probes_S[0])
        self.output = (voltage_probes, spike_probes)
        return self.output
