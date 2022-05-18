import numpy as np
from numpy import linalg
import networkx as nx
from networkx.algorithms.distance_measures import center


import matplotlib.pyplot as plt


import itertools

def pack_options(S, A):
    options = []
    for i in range(1,len(S)):
        option = (S[0], S[i])
        options.append(option)

        A[option[0],option[1]] = 1
        A[option[1],option[0]] = 1

    return options


"""
https://www.mlgworkshop.org/2019/papers/MLG2019_paper_41.pdf
"""
def ApproxAverageOptions(G,k):
    A = G.copy()
    graph = nx.to_networkx_graph(A)

    D_dict = nx.all_pairs_shortest_path_length(nx.to_networkx_graph(A))
    D = np.zeros(A.shape,dtype='int')
    for source in D_dict:
        for target in source[1].keys():
            D[source[0],target] = source[1][target]

    def cost(S):
        sum = 0
        for i in range(len(A)):
            shortest_dist = D[S[0],i]
            for j in range(1,len(S)):
                shortest_dist = min(shortest_dist, D[S[j],i])
            sum += shortest_dist
        return sum


    def get_trees(S):
        paths = nx.multi_source_dijkstra_path(graph, list(S))
        A_ = np.zeros(A.shape)
        for i in paths.keys():
            path = paths[i]
            for j in range(0,len(path)-1):
                A_[path[j],path[j+1]] = 1
        components = nx.connected_components(nx.to_networkx_graph(A_))
        trees = [graph.subgraph(c).copy() for c in components]
        return trees

    S = np.random.choice(len(A), k+1, replace=False)

    min_cost = cost(S)
    while True:
        T_list = get_trees(S)
        S_ = np.array([], dtype='int')
        for T in T_list:
            S_ = np.append(S_, center(T)[0])
        cost_new = cost(S_)
        S = S_
        if cost_new >= min_cost:
            break
        min_cost = cost_new


    options = pack_options(S, A)
    return A, options
    # options = []
    # for i in range(len(S)-1):
    #     best_j = None
    #     min_cost = None
    #     for j in range(i+1,len(S)):
    #         c = D[S[i],S[j]]
    #         if best_j == None:
    #             best_j = j
    #             min_cost = c
    #             continue
    #         if c > min_cost:
    #             best_j = j
    #             min_cost = c

    #     option = (S[i], S[best_j])
    #     options.append(option)

    #     A[option[0],option[1]] = 1
    #     A[option[1],option[0]] = 1

    # return A, options


def test():
    N = 100
    Gnx = nx.cycle_graph(N)
    #Gnx = nx.path_graph(N)

    #Gnx = nx.random_regular_graph(d=2, n=N)
    #Gnx = nx.barabasi_albert_graph(n=N,m=1)
    A = nx.to_numpy_matrix(Gnx).astype(dtype='int')

    A_, options = ApproxAverageOptions(A, 1)
    print(options)


