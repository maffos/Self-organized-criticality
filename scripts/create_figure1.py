"""
This script reproduces Figure 1 in Levina et al. (2007).

References:
    Levina, Anna, J. Michael Herrmann, and Theo Geisel. "Dynamical synapses causing self-organized criticality in neural networks." Nature physics 3.12 (2007): 857-860.
"""


import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np

alphas = [0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0]
N = 300
duration = 2000000
tau = 10
path = '../Data/N%d_%ds' % (N, duration/tau)

#run the model for each value of alpha
for alpha in alphas:
        os.system("python run_single_simulation.py {} {} {} -p {}".format(alpha,N, duration, path))

for alpha in alphas:

    filename = os.path.join(path, 'alpha{}_distribution.csv'.format(alpha))
    if alpha < 1.3:
        c = 'green'
        opac = .6
    elif alpha < 1.7:
        c = 'red'
        opac = .7
    else:
        c = 'blue'
        opac = .3

    df = pd.read_csv(filename)
    order = np.argsort(df.L)
    plt.plot(df.L[order], df.P_L[order], '--', c = c, alpha = opac)

blue_line = mlines.Line2D([], [], color='blue', linestyle = '--', label=r'$\alpha<2.1$')
red_line = mlines.Line2D([], [], color='red', linestyle='--', label=r'$\alpha<1.6$')
green_line = mlines.Line2D([], [], color='green', linestyle='--', label=r'$\alpha<1.2$')
plt.legend(handles= [blue_line,red_line,green_line])
plt.xscale('log')
plt.yscale('log')
plt.xlabel('L')
plt.ylabel('P(L)')
plt.show()
