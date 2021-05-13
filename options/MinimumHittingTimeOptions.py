import numpy as np
from numpy import linalg
import networkx as nx
from networkx.algorithms.distance_measures import center
from scipy.optimize import linprog
from scipy.spatial import ConvexHull

import matplotlib.pyplot as plt
import itertools
import time

from line_profiler import LineProfiler

"""
https://arxiv.org/pdf/1709.01633.pdf
"""

time_dict = {}
RAND_POINTS = None

NUM_RAND_POINTS = 128000

def log_time(tag, t):
    global time_dict
    if not tag in time_dict.keys():
        time_dict[tag] = []
    time_dict[tag].append(t)

def print_times():
    global time_dict
    for key in time_dict.keys():
        print(key, np.mean(time_dict[key]))

def MinimumHittingOptions(G, S, k):

    n = len(G)

    z_max = 20.0
    z_min = 0.0

    global RAND_POINTS, NUM_RAND_POINTS
    NUM_RAND_POINTS = 2 ** n
    RAND_POINTS = np.random.rand(NUM_RAND_POINTS, n)

    while X(S, 5, G) == 0:
        NUM_RAND_POINTS *= 2
        RAND_POINTS = np.random.rand(NUM_RAND_POINTS, n)
        print(NUM_RAND_POINTS)

    NUM_RAND_POINTS *= 2
    RAND_POINTS = np.random.rand(NUM_RAND_POINTS, n)
    print(NUM_RAND_POINTS)



    while z_max - z_min > .01:
        z = (z_max + z_min) / 2.0
        P = np.copy(G)
        options = []
        print(z)
        while X(S, z, P) > 0 and len(options) <= k:
            t1 = time.time()
            best_o = None
            best_o_X = np.Inf
            for i in range(n):
                for j in range(n):
                    if i == j or G[i,j] == 1:
                        continue
                    t2 = time.time()
                    X_val = X(S, z, P, (i,j))
                    if best_o == None or X_val < best_o_X:
                        best_o_X = X_val
                        best_o = (i,j)
                    log_time('eval one option',time.time()-t2)
            log_time('finding best option',time.time()-t1)
            options.append(best_o)
            P[best_o[0],best_o[1]] = 1
        if len(options) <= k:
            z_max = z
        else:
            z_min = z

    return P, options

def X(S, z, P_O, O=None):
    t = time.time()

    global RAND_POINTS, NUM_RAND_POINTS

    if O != None:
        temp = P_O[O[0],O[1]]
        P_O[O[0],O[1]] = 1

    v = np.zeros((len(P_O),))
    delta = delta_to_valid(S, v, P_O)
    while np.sum(delta) != 0 and valid(S, v, P_O):
        delta = delta_to_valid(S, v, P_O)
        v += delta

    log_time('X_func prep',time.time()-t)
    t = time.time()
    hull = ConvexHull(RAND_POINTS)
    print(hull.volume)
    # points = RAND_POINTS * v
    # def count(x):
    #     return np.array([np.sum(x) > z and valid(S, x, P_O)])
    # rez = np.apply_along_axis(count, 1, points)
    # count = np.sum(rez)
    count = 1
    log_time('X_func point sampling',time.time()-t)

    if O != None:
        P_O[O[0],O[1]] = temp
    return np.prod(v[np.nonzero(v)]) * count / NUM_RAND_POINTS

def valid(S, v, P_O):
    if np.sum(v[S]) != 0:
        return False
    for i in range(len(v)):
        if v[i] > 1 + np.min(v[np.where(P_O[i] != 0)[0]]):
            return False
    return True

def delta_to_valid(S, v, P_O):
    delta = np.zeros(v.shape)
    for i in range(len(v)):
        delta[i] = -v[i] + 1 + np.min(v[np.where(P_O[i] != 0)[0]])
    delta[S] = 0
    return delta


def test():
    N = 12
    #Gnx = nx.cycle_graph(N)
    #Gnx = nx.path_graph(N)

    Gnx = nx.random_regular_graph(d=3, n=N)
    #Gnx = nx.barabasi_albert_graph(n=N,m=1)
    A = nx.to_numpy_matrix(Gnx).astype(dtype='int')
    print(A)
    start_time = time.time()
    A_, options = MinimumHittingOptions(A, [0], 2)
    print("time: ", time.time()-start_time)
    print(A_, options)

    pos = nx.spectral_layout(Gnx)

    fig, axs = plt.subplots(1,2)
    nx.draw(nx.to_networkx_graph(A),ax=axs[0], pos=pos, node_size=25, node_color=(1,0,0))
    nx.draw(nx.to_networkx_graph(A_, create_using=nx.DiGraph),ax=axs[1], pos=pos, node_size=25, node_color=(1,0,0), with_labels=True)
    plt.show(block=True)


#
test()
print_times()
