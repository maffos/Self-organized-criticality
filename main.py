from brian2 import *
import numpy as np
import pandas as pd
import os
import testing
import argparse

def build_and_run(duration = 10000*ms, N=300, alpha = 1.4, u = 0.2, tau = 10*ms, tau_d = 1*ms, nu = 10, I = 0.025, theta = 1, topology = 'fc', dt = 1*ms, discrete = True, initialisation = {'h': 'uniform', 'J': 'uniform'}, random_state = 1, event_driven = True, p = None, record_states = True, plot_results = True, filename = None):

    np.random.seed(random_state)
    start_scope()
    
    if discrete: #sets the timestep to tau which makes the model discrete
        dt = tau

    #All dynamics of the model are event-based e.g. spike-trigered or given by external drive. Thus we only need to specify the variable for the membrane potential but no further dynamics.
    eqs = "h:1"
          
    G = NeuronGroup(N, eqs, threshold = 'h>=theta', reset = 'h -= theta', method = 'euler', dt = dt)
    
    #initialise mebrane potentials
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
    #the dynamics of the synaptic strength J are implemented in the Synapse class as well as the spike triggered postsynaptic potential
    #the decrement of the synaptic strength at spike time needs to be adjusted with a correction term, as the governing ODE as specified in the equation dJ/dt gets executed first. Thus we cannot simply write J-=u*J. As in this case J has already been updated.
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
         
    return [neuron_state,synapse_state], spikes
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("alpha", help= "Specify the value for the critical parameter alpha.", type =float)
    parser.add_argument("duration", help= "Specify the length of the duration in ms.", type =int)
    parser.add_argument("-t", "--test", help="Run tests before simulation.",action="store_true")
    parser.add_argument("-r", "--random", help = "Set the random state used in the Simulation", type = int)
    parser.add_argument("-s", "--save", help = "Set to save the data to disk.", action = "store_true")
    parser.add_argument("-p", "--plot", help = "Set to plot raster plot.", action = "store_true")
    args = parser.parse_args()

    if args.test:
        testing.run_all_tests()
    if args.random:
        random_state = args.random
    else:
        random_state = 1 
        
    duration = args.duration*ms
    tau = 10*ms

    N = 300
    path = 'Data/N237_%ds/%d'%(duration/ms/1000, random_state)
    if args.save:
        filename = os.path.join(path, 'alpha{}.csv'.format(args.alpha))
    else:
        filename = None
        
    statemonitors, spikemonitor = build_and_run(duration, N, alpha=args.alpha, random_state = random_state, initialisation = {'h': 'uniform', 'J':1}, tau=tau, dt = tau, plot_results = args.plot, record_states = False, filename = filename)
    
