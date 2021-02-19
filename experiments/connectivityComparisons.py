import numpy as np
from numpy import linalg
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
from options.graph.cover_time import AddEdge, ComputeCoverTime
from options.graph.spectrum import ComputeFiedlerVector, ComputeConnectivity

#matplotlib.use("TkAgg")
matplotlib.style.use('default')
"""
IN PAPER:
We randomly generated shortest path problems and plotted
the relationship between the value of a random policy, the
cover time, and the algebraic connectivity of the state-space
graph

TODO:
Generate graphs similarly and plot against Covertime, algebraic connectivity

Plots:

ASPD versus algebraic connectivity #same as old
ASPD versus random policy cost #same as old
ASPD versus Covertime  # ASPD should correlate with covertime
"""

def get_cover_time(A):
    D_dict = nx.all_pairs_shortest_path_length(nx.to_networkx_graph(A))
    C = 0
    for source in D_dict:
        sum = 0
        for target in range(len(A)):
            sum += source[1][target]
        C = max(C,sum/len(A))
    return C

def get_random_policy_cost(A):
    return 0

def get_algebraic_connectivity(A):
    return nx.algebraic_connectivity(nx.to_networkx_graph(A))

# A - adjacency, W - weights for pairs of vertices
def average_shortest_distance(A):
    D_dict = nx.all_pairs_shortest_path_length(nx.to_networkx_graph(A))
    sum = 0
    for source in D_dict:
        for target in range(len(A)):
            sum += source[1][target]
    return sum/(len(A) * len(A))


#generates random graph as in paper
def generate_random_graph(N, connectivity):
    A = np.zeros((N,N),dtype='int')
    edges = 0
    #first make a random tree
    for i in range(1,N):
        j = np.random.randint(0,i)
        A[i,j] = 1
        A[j,i] = 1
        edges += 1

    while edges / N*N < connectivity:

        i = np.random.randint(0,N)
        j = np.random.randint(0,N)
        while A[i,j] == 0 and i != j:
            i = np.random.randint(0,N)
            j = np.random.randint(0,N)
        A[i,j] = 1
        A[j,i] = 1
        edges += 1
    return A


def plot_algebraic_versus_ASPD(samples, N = 10, connectivity = .3):
    X = []
    Y = []
    for i in range(samples):
        A = generate_random_graph(N, connectivity)
        W = np.ones((N,N))
        X.append(get_algebraic_connectivity(A))
        Y.append(average_shortest_distance(A))

    plt.title("algebraic connectivity versus average shortest distance")
    plt.scatter(X,Y, s=10)
    plt.xlabel('algebraic connectivity')
    plt.ylabel('average shortest distance')
    plt.show(block=True)

def plot_ASPD_versus_cover_time(samples, N = 10, connectivity = .3):
    X = []
    Y = []
    for i in range(samples):
        A = generate_random_graph(N, connectivity)
        W = np.ones((N,N))
        X.append(average_shortest_distance(A))
        Y.append(get_cover_time(A))

    plt.title("average shortest distance versus cover time")
    plt.scatter(X,Y, s=10)
    plt.xlabel('average shortest distance')
    plt.ylabel('cover time')
    plt.show(block=True)


plot_algebraic_versus_ASPD(samples=100)
plot_ASPD_versus_cover_time(samples=200,N=50)
