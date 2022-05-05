"""
This script runs a single simulation, keeping most parameters fixed to standard values. Only the critical parameter alpha, the network size and the length need to be specified. 
Optionally one can also select whether the data should be stored, select the random state and specify a path where to store data. 
"""

import soc
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("alpha", help= "Specify the value for the critical parameter alpha.", type =float)
parser.add_argument("N", help = "Specify the number of Neurons in the network.", type = int)
parser.add_argument("duration", help= "Specify the length of the duration in ms.", type =int)
parser.add_argument("-s", "--save", help = "Set to save the data to disk.", action = "store_true")
parser.add_argument("-r", "--random", help = "Specify the random state", type = int)
parser.add_argument("-p", "--path", help = 'Specify the path were Data is stored', type = str)
args = parser.parse_args()

if args.random:
    random_state = args.random
else:
    random_state = 1

if args.path:
    path = args.path
else:
    path = 'Data/N%d_%ds'%(args.N, args.duration)

if args.save:
    filename = os.path.join(path, 'alpha{}.csv'.format(args.alpha))
else:
    filename = None
    
statemonitors, spikemonitor = soc.build_and_run(args.duration, args.N, alpha=args.alpha, random_state = random_state, initialisation = {'h': 'uniform', 'J':1}, plot_results = False, record_states = False, filename = filename)
soc.avalanche_distribution(spikemonitor,path, alpha = args.alpha, write_to_disk = True)
