import seaborn as sns
import matplotlib.pyplot as plt
from brian2 import *
from collections import Counter
import numpy as np
import os
import pandas as pd
import matplotlib.lines as mlines
            
def find_max_number_avalanches(alpha, offset, tau = 10, color = 'b', show_plot = True, random_state = 1):

    path = 'Data/sub1/2000s/%d'%random_state
    filename = os.path.join(path, 'alpha{}.csv'.format(alpha))
    data = pd.read_csv(filename)
    spike_times = np.sort(data.t)
    intervalls = np.diff(spike_times[offset:])
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
    
    if show_plot:
        plt.plot(spike_times[:offset], data.i[:offset], '.k')
        plt.show()
        
    max_num = np.sum(list(avalanche_sizes.values()))
    print('Max Number of avalanches is ', max_num)
    print('Offset is: ', offset)
    
    return max_num

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

def avalanche_distribution_initial_conditions(path, alpha, random_states, offset = None, create_plot = True, show_plot = True, write_to_disk = False):

    size_df = {'L': [], 'P_L': []}
    length_df = {'L':[], 'P_L':[]}
    size_counter = Counter()
    length_counter = Counter()
    for random_state in random_states:
        dir_path = os.path.join(path,str(random_state))
        if len(os.listdir(dir_path)) == 0:
            try:
                os.rmdir(dir_path)
            except OSError as e:
                print("Error: %s : %s" % (dir_path, e.strerror))
        else:
            #filename = os.path.join(path, str(random_state), "alpha{}.csv".format(alpha))
            size_update, length_update = avalanche_distribution(path = dir_path, alpha = alpha, offset = offset, show_plot = False)
            L = np.array(list(size_update.keys()))
            P_L = np.array(list(size_update.values()))/np.sum(list(size_update.values()))
            for key,value in zip(L,P_L):
                size_df['L'].append(key)
                size_df['P_L'].append(value)
                
            L = np.array(list(length_update.keys()))
            P_L = np.array(list(length_update.values()))/np.sum(list(length_update.values()))
            for key,value in zip(L,P_L):
                length_df['L'].append(key)
                length_df['P_L'].append(value)
            size_counter.update(size_update)
            length_counter.update(length_update)
         
    size_df = pd.DataFrame(size_df)
    length_df= pd.DataFrame(length_df)
    
    if create_plot: 
        sns.set_style("whitegrid")
        sns.despine()
        plt.scatter(size_df['L'], size_df['P_L'], s = 80, facecolors = 'none', edgecolors = 'black')
        mean = size_df.groupby('L').mean()
        plt.plot(mean.index, mean['P_L'], '--', c='blue', label = 'Mean')
        plt.title(r'$\alpha$={}'.format(alpha))
        plt.xscale('log')
        plt.yscale('log')
        plt.legend()
        if show_plot:
            plt.show()
        
    if write_to_disk:
        outfile_seq_size = os.path.join(path, 'alpha{}_sequence.csv'.format(alpha))
        outfile_seq_length = os.path.join(path, 'alpha{}_sequence_length.csv'.format(alpha))
        outfile_dist_size = os.path.join(path, 'alpha{}_distribution.csv'.format(alpha))
        outfile_dist_length = os.path.join(path, 'alpha{}_distribution_length.csv'.format(alpha))

        #write the avalanche size sequence       
        with open(outfile_seq_size, 'w') as f:
            for elem in size_counter.elements():
                f.write(str(elem) + '\n') 
                
        #write the distribution of avalanche sizes
        size_df.to_csv(outfile_dist_size)
                
        #write the sequence of avalanche lengths
        with open(outfile_seq_length, 'w') as f:
            for elem in length_counter.elements():
                f.write(str(elem) + '\n') 
                
        #write the distribution of avalanche lengths
        length_df.to_csv(outfile_dist_length)
        
    return size_df, length_df, size_counter, length_counter

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


def create_fig1_initial_conditions(path, random_states, offsets = None):

    sns.set_theme(rc={'figure.figsize':(11,6)}, style = 'whitegrid')
    sns.despine()
    alphas = [0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0]
    for alpha in alphas:
    
        if alpha < 1.2:
            c = 'green'
            opac = .6
        elif alpha < 1.6:
            c = 'red'
            opac = .7
        else:
            c = 'blue'
            opac = .3
            
        if offsets is not None:
            offset = offsets[alpha]
        else:
            offset = 0
            
        size_df, length_df, size_counter, length_counter = avalanche_distribution_initial_conditions(path, alpha, random_states, offset = offset, create_plot = False, show_plot = False)
        mean = size_df.groupby('L').mean()
        plt.plot(mean.index, mean['P_L'], '--', c=c, alpha = opac)
        
    blue_line = mlines.Line2D([], [], color='blue', linestyle = '--', label=r'$\alpha<2.1$')
    red_line = mlines.Line2D([], [], color='red', linestyle='--', label=r'$\alpha<1.6$')
    green_line = mlines.Line2D([], [], color='green', linestyle='--', label=r'$\alpha<1.2$')
    plt.legend(handles= [blue_line,red_line,green_line])
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('L')
    plt.ylabel('p(L)')
    plt.show() 

        
if __name__ == '__main__':
    path = '/home/matthias/Uni/WS2122/CSN/Project/Data/N300_2000s'
    random_states = np.arange(2,11)
    create_fig1_initial_conditions(path, random_states = random_states)

