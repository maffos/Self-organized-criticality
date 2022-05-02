import seaborn as sns
import matplotlib.pyplot as plt
from brian2 import *
from collections import Counter
import numpy as np
import os
import pandas as pd
import matplotlib.lines as mlines

def avalanche_distribution(data = None, path = None, alpha = None, tau = 10, offset = 0, write_to_disk = False, show_plot = True, color = 'b'):
    
    if data is None:
        filename = os.path.join(path,"alpha{}.csv".format(alpha))
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
        plt.plot(L[order], P_L[order], '--', c = color)
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('L')
        plt.ylabel('P(L)')
        plt.show()
        
    if write_to_disk:
    
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
    return avalanche_sizes, avalanche_lengths

def create_fig1(path):


    alphas = [0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0]
    for alpha in alphas:

        filename = os.path.join(path, 'alpha{}.csv'.format(alpha))
        if alpha < 1.3:
            c = 'green'
        elif alpha < 1.6:
            c = 'red'
        else:
            c = 'blue'

        L, P_L = avalanche_distribution(filename = filename, show_plot = False, color = c)
        plt.plot(L, P_L, '--', c = c)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('L')
    plt.ylabel('P(L)')
    plt.show()

