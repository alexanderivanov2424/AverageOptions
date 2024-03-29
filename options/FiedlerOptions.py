import numpy as np
from numpy import linalg
import networkx as nx
import matplotlib.pyplot as plt
from options.graph.utils import AddEdge, ComputeCoverTime
from options.graph.spectrum import ComputeFiedlerVector, ComputeConnectivity


def FiedlerOptions(G, k, subgoal=False):
    no = 0

    X = nx.to_networkx_graph(G)
    if not nx.is_connected(X):
        cs = list(nx.connected_components(X))
        for c_ in cs:
            if len(c_) > 1:
                c = c_
                break
        Xsub = X.subgraph(c)
        A = nx.to_numpy_matrix(Xsub)
        print('connected comp =', c)
    else:
        A = G.copy()

    options = []

    eigenvalues = []
    eigenvectors = []

    while no < k:
        v = ComputeFiedlerVector(nx.to_networkx_graph(A))
        #lmd = ComputeConnectivity(A) #NOT BEING USED

        maxs = [np.argmax(v)]
        mins = [np.argmin(v)]
        option = (maxs, mins)

        options.append(option)
        if subgoal:
            B = A.copy()
            B[:, option[1][0]] = 1
            B[option[1][0], :] = 1
        else:
            def AddEdge(G, vi, vj):
                augGraph = G.copy()
                # print('augGraph', augGraph)
                augGraph[vi, vj] = 1
                augGraph[vj, vi] = 1
                return augGraph
            B = AddEdge(A, option[0][0], option[1][0])
        A = B
        no += 2
        # eigenvalues.append(lmd) #NOT BEING USED
        eigenvectors.append(v)

    # TODO: If A is a subgraph of G, convert the acquired eigenvectors to the original size.
    if not nx.is_connected(X):
        evecs = []
        for v in eigenvectors:
            newv = np.zeros(G.shape[0])
            i = 0
            j = 0
            while i < A.shape[0]:
                if j in c:
                    newv[j] = v[i]
                    i += 1
                j += 1
            evecs.append(newv)
    else:
        evecs = eigenvectors

    return A, options, eigenvalues, evecs

if __name__ == "__main__":

    Gnx = nx.path_graph(10)

    #Gnx = nx.cycle_graph(30)
    graph = nx.to_numpy_matrix(Gnx)
    print('#'*10)
    for i in range(5):
        t = ComputeCoverTime(graph)
        print('Number of Options',i)
        print('CoverTime     ', t)
        lb = nx.algebraic_connectivity(nx.to_networkx_graph(graph))
        print('lambda        ', lb)
        print()
        graph, options, _, _ = FiedlerOptions(graph, 1)
        print(options)
    # proposedAugGraph, options, _, _ = FiedlerOptions(graph, 8)
    #
    # pGnx = nx.to_networkx_graph(proposedAugGraph)
    #
    # nx.draw_spectral(pGnx)
    # plt.savefig('drawing.pdf')
    #


    # t = ComputeCoverTime(graph)
    # print('CoverTime     ', t)
    # lb = nx.algebraic_connectivity(nx.to_networkx_graph(graph))
    # print('lambda        ', lb)
    #
    # t3 = ComputeCoverTime(proposedAugGraph)
    # print('CoverTime Aug ', t3)
    # lb3 = nx.algebraic_connectivity(nx.to_networkx_graph(proposedAugGraph))
    # print('lambda        ', lb3)
