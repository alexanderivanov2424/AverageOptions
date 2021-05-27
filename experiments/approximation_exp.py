from options.AverageOptions import AverageShortestOptions, BruteOptions
from options.ApproxAverageOptions import ApproxAverageOptions

from options.OptionGeneration import GetOptions

import numpy as np
from numpy import linalg
import networkx as nx
import scipy.stats

import matplotlib
import matplotlib.path as mpath
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

def average_shortest_distance(A):
    D_dict = nx.all_pairs_shortest_path_length(nx.to_networkx_graph(A))
    sum = 0
    for source in D_dict:
        for target in range(len(A)):
            sum += source[1][target]
    return sum

def get_approx_ratios(G_list, brute_vals, method):
    ratios = []
    for G, brute_val in zip(G_list,brute_vals):
        print(len(G))
        G_, ops, _ = GetOptions(G, 1, method, verbose=False)
        print(ops)
        ratios.append(average_shortest_distance(G_) / brute_val)

    return ratios



X = [10, 20, 30, 40, 50, 60]#, 40, 50]
graph_list = [nx.fast_gnp_random_graph(N, .3) for N in X]
G_list = [nx.to_numpy_matrix(graph.subgraph(max(nx.connected_components(graph), key=len)).copy())for graph in graph_list]
brute_vals = []
for G in G_list:
    N = len(G)
    print(N)
    P = np.array([np.random.permutation(np.arange(N))[:2] for i in range(N*N)])
    G_, ops = BruteOptions(G,P, 1)
    print(ops)
    brute_vals.append(average_shortest_distance(G_))

ASPDM_list = get_approx_ratios(G_list, brute_vals, 'ASPDM')
approx_list = get_approx_ratios(G_list, brute_vals, 'ApproxAverage')

plt.plot(X, ASPDM_list, c='tab:green', label='ASD Options')
plt.plot(X, approx_list, c='tab:red', label='Fast ASD Options')
plt.title("Approximation Ratio for ASD and Fast ASD")
plt.xlabel('number of vertices')
plt.ylabel('approximation ratio')
plt.legend()
plt.show()
print(ASPDM_list)
