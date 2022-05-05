from brian2 import *
import numpy as np
import pandas as pd
import os
from collections import Counter
import seaborn as sns

def build_and_run(duration = 10000*ms,
                  N=300,
                  alpha = 1.4,
                  u = 0.2,
                  tau = 10*ms,
                  tau_d = 1*ms,
                  nu = 10,
                  I = 0.025,
                  topology = 'fc',
                  dt = 1*ms,
                  discrete = True,
                  initialisation = {'h': 'uniform', 'J': 'uniform'},
                  random_state = 1,
                  event_driven = True,
                  p = None,
                  record_states = False,
                  plot_results = True,
                  filename = None):

    """
    Main method. Constructs a model of self-organized-criticality according to the passed parameters and runs the simulation.

    Args:
        duration   (brian2 unit, int, float)   :   Length of the simulation
        N  (int)   :   Number of Neurons in the Network.
        alpha  (float) :   Value of the critical parameter.
        u  (float) :   Fraction by which synaptic strenth gets reduced.
        tau (brian2 unit, int, float)   :   Time scale of the driving stochastic process.
        tau_d   (brian2 unit, int,float)    :   Synaptic timescale.
        nu  (float) :   Parameter to determine the timescale of the dynamics of the synaptic strength.
        I   (float) :   Magnitude of the external drive.
        topology   (str, tuple)    :    Specifies the network topology either by a string or by a tuple containing an array of presynaptic indices in the first entry and an array of corresponding postsynaptic indices in the second entry.
        dt  (brian2 unit, int, float)   :   The temporal resolution of the numeric solver.
        discrete    (bool)  :   Select, whether temporal integration is done in a discrete way or continously.
        initialisation  (dict): Dictionary of the form {'h': (string, numerical, array), 'J': (str, numerical, array)}. Initialises the membrane potentials and the synaptic strengths.
        random_state    (int)   :   Choose the random state
        event_driven    (bool)  :   If selected, brian2 will update the synaptic strengths only upon spiking event which saves computation.
        p   (float) :   Probability under which a Synapse is constructed if 'random' is selected as topology.
        record_states   (bool)  :   If selected, brian2 will record the membrane potential and synaptic strengths. This can get very memory intensive.
        plot_results    (bool)  : If selected, a raster plot will be plotted after simulation as well as an exemplary neuron trace, if record_states is selected.
        filename    (str)   :   If specified, the data of the brian2 spike monitor will get stored to the specified file.

    Returns:
        (neuron_state,synapse_state)    :   Tuple containing the brian2 monitors of the membrane potential and synaptic strengths. Only contain data if record_states is selected.
        spikes  :   Spike Data collected in the Brian2.SpikeMonitor class.
    """

    np.random.seed(random_state)
    start_scope()
    #define constants
    THETA = 1
    #if variables that require physical units have been passed as numerics, convert them to units.
    def to_unit(variable, unit):
        if isinstance(variable, (int, float)):
            variable = variable * unit
        return variable

    duration = to_unit(duration, ms)
    tau = to_unit(tau, ms)
    tau_d = to_unit(tau_d, ms)
    dt = to_unit(dt, ms)

    if discrete: #sets the timestep to tau which makes the model discrete
        dt = tau

    #All dynamics of the model are event-based e.g. spike-trigered or given by external drive. Thus we only need to specify the variable for the membrane potential but no further dynamics.
    eqs = "h:1"
          
    G = NeuronGroup(N, eqs, threshold = 'h>=THETA', reset = 'h -= THETA', method = 'euler', dt = dt)
    
    #initialise membrane potentials
    if initialisation['h'] == 'uniform':
        G.h = np.random.sample(size = N)
    elif initialisation['h'] == 'zeros':
        G.h = 0
    elif isinstance(initialisation['h'], (int,float, np.ndarray)):
            G.h = initialisation['h']
    else:
        raise ValueError('The value "{}" is not supported for initialisation. Either use an int, float or numpy array or specify a string. In this case the string has to be either "uniform" or "zeros". '.format(initialisation['h'])) 
    
    #gets called by the brian2 NetworkOperation class, implements the external drive
    def external_input():
        idx = np.random.randint(N)
        G.h[idx]+=I
        
    #this calls the external input every tau steps
    net_op = NetworkOperation(external_input, dt= tau)
    
    #calculate the time-scale of the dynamics of the synaptic strength
    tau_j = tau*nu*N
    J_max = alpha/u
    N_neurons = N #we have to rename N because the class Synapses uses this variable internally, too


    #The dynamics of the synaptic strength J are implemented in the Synapse class as well as the spike triggered postsynaptic potential.
    #The decrement of the synaptic strength at spike time needs to be adjusted with a correction term, as the governing ODE as specified in the equation dJ/dt gets executed first. Thus we cannot simply write J-=u*J. As in this case J has already been updated.

    if event_driven:
        eqs =  "dJ/dt = 1/tau_j*(alpha/u-J) : 1 (event-driven)"
    else:
        eqs =  "dJ/dt = 1/tau_j*(alpha/u-J) : 1"
        
    S = Synapses(G,G, eqs, on_pre = {'pre_transmission': 'h_post += u/N_neurons*J',
                                     'pre_selforg': 'J = clip(J-u*((J-alpha/(tau_j/ms*u))/(1-1/(tau_j/ms))), 0, J_max)'}, dt = dt, method = 'euler')
    #fully-connected network
    if topology == 'fc':                                                                             
        S.connect(condition = 'i!=j')
    #if a random connection of the nodes is preferred, p has to be set to a value between 0 and 1
    elif topology == 'random':
        S.connect(condition = 'i!=j', p = p)
    #a tuple is passed as the network topology, containing the coordinates of the non-zero entries of the adjacency matrix
    elif isinstance(topology, tuple):
        S.connect(i=topology[0], j=topology[1]) 

    #initialise synaptic strength
    if initialisation['J'] == 'uniform':
        S.J = np.random.uniform(0,J_max, size = S.J.shape[0])
    elif initialisation['J'] == 'zeros':
        S.J = 0
    elif isinstance(initialisation['J'], np.ndarray):
        if len(S.J) < len(initialisation['J']):
            S.J = initialisation['J'][:len(S.J)]
        elif len(S.J) > len(initialisation['J']):
            S.J[:len(initialisation['J'])] = initialisation['J']
            S.J[len(initialisation['J']):] = np.mean(initialisation['J'])
    elif isinstance(initialisation['J'], (int,float)):
            S.J = initialisation['J']
    else:
        raise ValueError('The value "{}" is not supported for initialisation. Either use an int, float or numpy array or specify a string. In this case the string has to be either "uniform" or "zeros". '.format(initialisation['J'])) 

       
    #specify the synaptic timescale that affects the membrane potential dynamics
    S.pre_transmission.delay = tau_d
 
    #specify the monitors which are later used for plotting
    spikes = SpikeMonitor(G)
    neuron_state = StateMonitor(G, 'h', dt = dt, record = record_states)
    synapse_state = StateMonitor(S, 'J', dt=dt, record = record_states)

    print('Running simulation...')
    run(duration)
    print('Done')
    
    #immediately plot sum results
    if plot_results:
        plt.plot((spikes.t/ms), spikes.i, '.k')
        plt.xlabel('Time(ms)')
        plt.ylabel('Neuron')

        plt.show()
        
        if record_states:
            plt.plot(neuron_state.t/ms, neuron_state.h[1], '-b')
            plt.xlabel('Time(ms)')
            plt.ylabel('h')
            plt.show()
       
    if filename:
        dirs = filename.split('/')
        dirs = dirs[:-1]
        dirs = '/'.join(dirs)
        if not os.path.isdir(dirs):
            os.makedirs(dirs)
        #only save spikedata
        df = pd.DataFrame({'t': spikes.t/ms, 'i': np.array(spikes.i)})
        df.head()
        df.to_csv(filename)
         
    return (neuron_state,synapse_state), spikes
   
#calculate the avalanche distribution from spike train data. Either passed as a saved csv file or as a brian2 spike monitor
def avalanche_distribution(data = None, path = None, suffix = None, alpha = None, tau = 10, offset = 0, write_to_disk = False, show_plot = True):

    """
    Calculates the avalanche distribution from spike train data, either passed as a brian2.SpikeMonitor or as specified in a file.

    Args:
    :param data: If None, a file name has to be passed, where the data is stored. Otherwise a Brian2 Spike Monitor.
    :param path: Path where data is stored and will be written.
    :param suffix: End of the file name that contains the data.
    :param alpha:   Value of the critical parameter alpha
    :param tau: Value of the temporal resolution. Used to bin the spike trains.
    :param offset:  If specified, Avalanche counting will start at the specified offset.
    :param write_to_disk:   If specified, values of the avalanche distribution will be written to disk.
    :param show_plot:   If specified the distribution will be plotted.
    """

    if data is None:
        filename = os.path.join(path,suffix)
        data = pd.read_csv(filename)
        spike_times = np.sort(data.t)
        intervalls = np.diff(spike_times[offset:])
    else:
        intervalls = np.diff(np.sort(data.t/ms))
        
    avalanche_sizes = Counter()
    avalanche_lengths = Counter()
    size = 1
    length = 1
    
    for i in range(len(intervalls)):
        if intervalls[i] <= tau:
            size += 1
            length += np.rint(intervalls[i]/tau)
        else:
            avalanche_sizes.update([size])
            avalanche_lengths.update([length])
            size = 1
            length = 1
    avalanche_sizes.update([size])
    avalanche_lengths.update([length])
    
    if show_plot:
        L = np.array(list(avalanche_sizes.keys()))
        P_L = np.array(list(avalanche_sizes.values()))/np.sum(list(avalanche_sizes.values()))
        order = np.argsort(L)
        sns.set_style("whitegrid")
        sns.despine()
        plt.plot(L[order], P_L[order], '--', c = 'b')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('L')
        plt.ylabel('P(L)')
        plt.show()
        
    if write_to_disk:

        if not os.path.exists(path):
            os.makedirs(path)
        outfile_seq_size = os.path.join(path, 'alpha{}_sequence.csv'.format(alpha))
        outfile_seq_length = os.path.join(path, 'alpha{}_sequence_length.csv'.format(alpha))
        outfile_dist_size = os.path.join(path, 'alpha{}_distribution.csv'.format(alpha))
        outfile_dist_length = os.path.join(path, 'alpha{}_distribution_length.csv'.format(alpha))

        #write the avalanche size sequence       
        with open(outfile_seq_size, 'w') as f:
            for elem in avalanche_sizes.elements():
                f.write(str(elem) + '\n') 
                
        L = np.array(list(avalanche_sizes.keys()))
        P_L = np.array(list(avalanche_sizes.values()))/np.sum(list(avalanche_sizes.values()))
        size_df = pd.DataFrame({'L': L, 'P_L': P_L})
        #write the distribution of avalanche sizes
        size_df.to_csv(outfile_dist_size)
                
        #write the sequence of avalanche lengths
        with open(outfile_seq_length, 'w') as f:
            for elem in avalanche_lengths.elements():
                f.write(str(elem) + '\n') 
                
        L = np.array(list(avalanche_lengths.keys()))
        P_L = np.array(list(avalanche_lengths.values()))/np.sum(list(avalanche_lengths.values()))
        length_df = pd.DataFrame({'L': L, 'P_L': P_L})
        #write the distribution of avalanche lengths
        length_df.to_csv(outfile_dist_length)
        
