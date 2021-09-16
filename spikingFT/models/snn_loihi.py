#!/usr/bin/env python3
"""
Module containing a class for implementing the S-FT in Loihi
"""
# Standard libraries
import logging
import numpy as np
import pathlib
# Local libraries
try:
    import nxsdk.api.n2a as nx
except ImportError:
    logger.warn("Intel NxSDK cannot be found. "
                "It will not be possible to run simulations with Loihi")
import spikingFT.models.snn

logger = logging.getLogger('spiking-FT')


class SNNLoihi(spikingFT.models.snn.FourierTransformSNN):
    """
    Class for setting up a network on Loihi for the spiking FT

    Attributes:
        nsamples (int): determined by number of samples in radar data;
         determines number of compartments in network
        sim_time (int): charging phase time of network in loihi time steps;
         spiking phase also takes sim_time
        net: loihi network object
        current_decay (int): sets current decay of  alls synapses
        l1_real_g: compartment group with nsamples compartments for the real
         coefficients of the DFT
        l1_imag_g: compartment group with nsamples compartments for the imag
         coefficients of the DFT
        real_weights: connection weights for "real" compartments
        imag_weights: connection weights for "imag" compartments
        board: loihi board which defines the compiler used
        bias_channel: channel for communicating bias data between ipynb and snip
        time_channel: channel for communicating time data between ipynb and snip
        l1_real_probes_V: voltage probes of "real" compartments
        l1_imag_probes_V: voltage probes of "imag" compartments
        l1_real_probes_S: spike probes of "real" compartments
        l1_imag_probes_S: spike probes of "imag" compartments
        et_probe: execution time probe
        e_probe: energy consumption probe
    """
    PLATFORM = "loihi"
    TH_MANT = 131071
    BIAS_EXP = 6
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

        # Initialize NETWORK and COMPARTMENTS
        self.net = nx.NxNet()
        self.l1_real_g, self.l1_imag_g = self.init_compartments()

        # SNIP
        self.board = None
        self.bias_channel = None
        self.time_channel = None

        # PROBES
        self.l1_real_probes_V = self.l1_real_g.probe(
                nx.ProbeParameter.COMPARTMENT_VOLTAGE,
                probeConditions=None
        )
        self.l1_imag_probes_V = self.l1_imag_g.probe(
                nx.ProbeParameter.COMPARTMENT_VOLTAGE,
                probeConditions=None
        )
        self.l1_real_probes_S = self.l1_real_g.probe(
                nx.ProbeParameter.SPIKE,
                probeConditions=None
        )
        self.l1_imag_probes_S = self.l1_imag_g.probe(
                nx.ProbeParameter.SPIKE,
                probeConditions=None
        )
        self.et_probe = None
        self.e_probe = None
        return


    def init_compartments(self):
        """
        Initializes compartments depending of the number of samples
        
        Returns:
            l1_real_g: compartment group with nsamples compartments for
             the real coefficients of the DFT
            l1_imag_g: compartment group with nsamples compartments for
             the imag coefficients of the DFT
        """
        core_distribution_factor = int(self.nsamples/64)
        # Real layer
        l1_real = []
        l1_real_g = self.net.createCompartmentGroup()
        for i in range(self.nsamples):
            l1_real_p = nx.CompartmentPrototype(
                vThMant=self.TH_MANT,
                biasMant=0,
                biasExp=self.BIAS_EXP,
                compartmentCurrentDecay=self.current_decay,
                compartmentVoltageDecay=0,
                functionalState=nx.COMPARTMENT_FUNCTIONAL_STATE.IDLE,
                logicalCoreId=int(i/core_distribution_factor),
                refractoryDelay=self.REFRACTORY_T
            )
            l1_real.append(self.net.createCompartment(prototype=l1_real_p))
        l1_real_g.addCompartments(l1_real)
        # Imaginary layer
        l1_imag = []
        l1_imag_g = self.net.createCompartmentGroup()
        for i in range(self.nsamples):
            l1_imag_p = nx.CompartmentPrototype(
                vThMant=self.TH_MANT,
                biasMant=0,
                biasExp=self.BIAS_EXP,
                compartmentCurrentDecay=self.current_decay,
                compartmentVoltageDecay=0,
                functionalState=nx.COMPARTMENT_FUNCTIONAL_STATE.IDLE,
                logicalCoreId=int(i/core_distribution_factor),
                refractoryDelay=self.REFRACTORY_T
            )
            l1_imag.append(self.net.createCompartment(prototype=l1_imag_p))
        l1_imag_g.addCompartments(l1_imag)
        return l1_real_g, l1_imag_g


    def connect_inputs(self, real_encoded_data, imag_encoded_data):
        """
        Generates N real and N imaginary input generators, N=nsamples
        
        Input generators are connected to the two the real and imaginary groups;
        the ttfs encoded radar data is given as input to the spike generators

        Parameters:
            real_encoded_data (array): dimensions (nsamples/1) ttfs encoding
            imag_encoded_data (array): dimensions (nsamples/1) ttfs encoding
        """
        logger.debug("Creating input spike generators")
        real_gen = self.net.createSpikeGenProcess(numPorts=self.nsamples)
        imag_gen = self.net.createSpikeGenProcess(numPorts=self.nsamples)
        in_l1 = nx.ConnectionPrototype(signMode=1, weightExponent=-3)
        # Connect input generators to the first layer of the SNN
        real_gen.connect(
            self.l1_real_g,
            prototype=in_l1,
            weight=self.real_weights
        )
        imag_gen.connect(
            self.l1_imag_g,
            prototype=in_l1,
            weight=self.imag_weights
        )
        # Specify spiking time of each generator
        logger.debug("Assigning spike times to input spike generators")
        for i in range(self.nsamples):
            real_gen.addSpikes(spikeInputPortNodeIds=i,
                               spikeTimes=[real_encoded_data[i]])
            imag_gen.addSpikes(spikeInputPortNodeIds=i,
                               spikeTimes=[imag_encoded_data[i]])
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

    def run(self):
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


    def simulate(self, data):
        # Create spike generators and connect them to compartments
        self.connect_inputs(data.real, data.real)
        # Instantiate measurement probes
        if self.measure_performance:
            network.performance_profiling()
        self.init_snip()
        # Run the network
        self.run()

        voltage_probes = (self.l1_real_probes_V[0], self.l1_imag_probes_V[0])
        spike_probes = (self.l1_real_probes_S[0], self.l1_imag_probes_S[0])
        self.output = (voltage_probes, spike_probes)
        return self.output
