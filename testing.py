from brian2 import *
from main import build_and_run
from numpy.testing import assert_equal, assert_almost_equal

def test_external_input():
    """
    This test runs a simulation of 1 Neuron for 15 ms, starting with h=0. At the beginning, the external input increases h by the amount of I.
    After tau=10ms, the external input should increase h by I again.
    """
     
    I =0.025
    tau = 10*ms
    duration = 15*ms
    N = 1
    dt = 1*ms
    initialisation = {'h': 0, 'J': 'uniform'}
    model = build_and_run(duration, N, I=I, tau=tau, dt = dt, plot_results = False)
    h = model.get_states()['statemonitor']['h']
    assert_equal(h[0], I)
    idx = int(tau/ms)
    assert_equal(h[idx], 2*I) 
    
def test_spike_event():
    """
    This test runs a simulation of 2 Neurons for 10ms. Neuron 0 is initialised with h=0 and Neuron 1 with h=1-I. The time constant of the external input is set to tau=5ms. Thus Neuron 1 should emit a spike at t=5ms, controlling the random seed.
    """
    I =0.025
    tau = 5*ms
    duration = 10*ms
    N = 2
    dt = 1*ms
    tau_d = 1*ms
    nu = 2
    u=0.2
    alpha = 1.4
    idx = int(tau/ms)
    transmission_idx = idx+int(tau_d/ms)+1
    initialisation = {'h': np.array([0,1-I]), 'J': 1}
    statemonitors,spikemonitor = build_and_run(duration, N, I=I, tau=tau, alpha = alpha, nu = nu, u=u, tau_d = tau_d, initialisation = initialisation, dt = dt, event_driven = False, random_state = 42, plot_results = False)
    h = statemonitors[0].h
    #J = model.get_states()['statemonitor_1']['J']
    J = statemonitor[1].J
    increment = 1/N*u*J[transmission_idx][1] 
    J_increment = 1/(tau/ms*nu*N)*(alpha/u-J[idx][1])
    J_decrement = u*J[idx][1]
    dJ_dt_0 = 1/(tau/ms*nu*N)*(alpha/u-J[0][1])
    spikes = model.get_states()['spikemonitor']
    assert_equal(h[idx][1], 1)
    print('...membrane potenial is 1 at spike time...')
    assert_equal(spikes['i'][0], 1)
    print('...spike recorded correctly in spike monitor...')
    assert_equal(spikes['t'][0], idx*ms)
    print('...time correct...')
    assert_equal(h[idx+1][1], 0)
    print('...reset of membrane potential correct...')
    assert_equal(h[transmission_idx][0], I+increment)
    print('...increase of postsynaptic membrane potential by correct amount...')
    assert_equal(h[transmission_idx-1][0], I)
    print('...transmission delay correct...')
    assert_almost_equal(J[1][1], J[0][1]+dJ_dt_0)
    print('...dynamics of synaptic strength when spike is absent are correct...')
    assert_almost_equal(J[idx+1][1], J[idx][1]+J_increment-J_decrement)
    print('...dynamics of synaptic strength after spike emission are correct...')
    
def run_all_tests():

    print('Testing external input...')
    test_external_input()
    print('Done')
    
    print('Test generation of a spike...')
    test_spike_event()
    print('Done')
    
    print('All tests passed!')
