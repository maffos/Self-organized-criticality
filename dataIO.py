import pandas as pd
import os
from collections import Counter
import numpy as np

def write_avalanche_sequence(path, value_range):

    for value in value_range:
    
        filename = os.path.join(path, 'alpha{}.csv'.format(value))
        outfile = os.path.join(path, 'alpha{}_sequence.csv'.format(value))
        df = pd.read_csv(filename)
        avalanches = Counter(df.t)
        with open(outfile, 'w') as f:
            for value in avalanches.values():
                f.write(str(value) + '\n')

def write_avalanche_distribution(path, value_range):
    for value in value_range:
    
        filename = os.path.join(path, 'alpha{}.csv'.format(value))
        outfile = os.path.join(path, 'alpha{}_distribution.csv'.format(value))
        df = pd.read_csv(filename)
        avalanches = Counter(df.t)
        
        avalanche_counts = Counter(avalanches.values()) 
        L = np.array(list(avalanche_counts.keys()))
        P_L = np.array(list(avalanche_counts.values()))/np.sum(list(avalanche_counts.values()))
        order = np.argsort(L)
        with open(outfile, 'w') as f:
            for L_i, P_Li in zip(L[order], P_L[order]):
                f.write(str(L_i) + ',' + str(P_Li) + ' \n')
                
def aggregate_avalanche_data(path, value_range, initial_conditions):

    for value in value_range:
        outfile_seq = os.path.join(path, 'alpha{}_sequence.csv'.format(value))
        outfile_seq_length = os.path.join(path, 'alpha{}_sequence_length.csv'.format(value))
        outfile_dist = os.path.join(path, 'alpha{}_distribution.csv'.format(value))
        outfile_dist_length = os.path.join(path, 'alpha{}_distribution_length.csv'.format(value))
        avalanche_sizes= Counter()
        avalanche_lengths = Counter()
        size_dist = {'L': [], 'P_L': []}
        length_dist = {'L':[], 'P_L':[]}
        #aggregate over initial conditions
        for initial_condition in initial_conditions:
            filename = os.path.join(path, str(initial_condition), 'alpha{}.csv'.format(value))
            df = pd.read_csv(filename)
            intervalls = np.diff(np.sort(data.t/ms))
            size = 1
            length = 1
            for i in range(len(intervalls)):
                if intervalls[i] <= tau:
                    size += 1
                    length += intervalls[i]/tau
                else:
                    avalanche_sizes.update([size])
                    avalanche_lengths.update([length])
                    size = 1
                    length = 1
            avalanche_sizes.update([size])
            avalanche_lengths.update([length])
        #write the avalanche sequence 
        with open(outfile_seq, 'w') as f:
            for elem in avalanche_sizes.elements():
                f.write(str(elem) + '\n') 
                
        #write the avalanche length sequence       
        with open(outfile_seq_size, 'w') as f:
            for elem in avalanche_lengths.elements():
                f.write(str(elem) + '\n') 
                
        #write the distribution
        L = np.array(list(avalanche_sizes.keys()))
        P_L = np.array(list(avalanche_sizes.values()))/np.sum(list(avalanche_sizes.values()))
        order = np.argsort(L)
        with open(outfile_dist, 'w') as f:
            for L_i, P_Li in zip(L[order], P_L[order]):
                f.write(str(L_i) + ',' + str(P_Li) + ' \n')
                
        #write the distribution of avalanche lengths
        L = np.array(list(avalanche_lengths.keys()))
        P_L = np.array(list(avalanche_lengths.values()))/np.sum(list(avalanche_lengths.values()))
        order = np.argsort(L)
        with open(outfile_dist_length, 'w') as f:
            for L_i, P_Li in zip(L[order], P_L[order]):
                f.write(str(L_i) + ',' + str(P_Li) + ' \n')
        
            

    
if __name__ == '__main__':
    path = 'Data/N300_2000s'
    alphas = [0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0]
    random_states = np.arange(10, dtype = int)+1
    aggregate_avalanche_data(path, alphas, random_states)
    #write_avalanche_sequence(path, alphas)
    #write_avalanche_distribution(path, alphas)
