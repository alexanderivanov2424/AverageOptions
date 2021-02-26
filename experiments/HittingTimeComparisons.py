from simple_rl.tasks import GridWorldMDP, TaxiOOMDP, HanoiMDP#, GymMDP
from simple_rl.tasks.grid_world.GridWorldMDPClass import make_grid_world_from_file
from simple_rl.tasks.race_track.RaceTrackMDPClass import make_race_track_from_file

from options.OptionClasses.Option import Option, getGraphFromMDP, constructOptionObject
from options.OptionClasses.OptionAgent import OptionAgent
from options.OptionGeneration import GetOptions

import matplotlib
import matplotlib.path as mpath
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

import numpy as np
import scipy.stats
import gc
import random
from tqdm import tqdm

import networkx as nx

matplotlib.style.use('default')

def get_brute_hitting_time_option(A):
    best_i_j = (0,0)
    best_hitting_time = None
    with tqdm(total=A.shape[0] * A.shape[1]) as pbar:
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                pbar.update(1)
                if A[i,j]:
                    continue
                B = A.copy()
                B[i,j] = 1
                B[j,i] = 1
                H = getHittingTime(B)
                if best_hitting_time == None:
                    best_hitting_time = H
                    best_i_j = (i,j)
                if best_hitting_time > H:
                    best_hitting_time = H
                    best_i_j = (i,j)

    B = A.copy()
    B[best_i_j[0], best_i_j[1]] = 1
    B[best_i_j[1], best_i_j[0]] = 1
    return B, best_i_j, None


def getHittingTimeTarget(A, t, N=1000, epsilon=.01):
    P = A / np.sum(A,axis=1)
    P[t,:] = 0
    # P[t,t] = 1
    H = np.zeros((A.shape[0],))
    I = np.eye(A.shape[0])
    for i in range(1, N):
        I = np.matmul(I, P)
        H += np.ravel(I[:,t]) * i
        if np.linalg.norm(P[:,t]) < epsilon:
            break
    return H

def getHittingTime(A):
    H = np.zeros(A.shape)
    for target in range(A.shape[0]):
        H[:,target] = getHittingTimeTarget(A, target)
    return np.mean(H)

def test_option_method(A, method="eigen"):
    print(method)
    X = []
    B = A.copy()
    X.append(getHittingTime(B))
    if method == "brute_hitting":
        for i in range(1,10):
            B, options, _ = get_brute_hitting_time_option(B)
            X.append(getHittingTime(B))
    else:
        for i in range(1,10):
            B, options, _ = GetOptions(A.copy(), i, method, verbose=False)
            print(options)
            X.append(getHittingTime(B))
    return X

def mulit_test(num_vertex, trials, method='eigen'):
    data = []
    for i in range(trials):
        Gnx = nx.fast_gnp_random_graph(n=num_vertex,p=.5, seed=i)
        Gnx = Gnx.subgraph(max(nx.connected_components(Gnx), key=len)).copy()
        A = nx.to_numpy_matrix(Gnx).astype(dtype='int')
        data.append(test_option_method(A, method=method))
    return data


color_dict = {  "eigen":"tab:blue",
                "fiedler":"tab:orange",
                "ASPDM":"tab:green",
                "ApproxAverage":"tab:red",
                "hitting":"cyan",
                "brute_hitting":"black",
                }

N = 20

for method in ["brute_hitting", "eigen", "fiedler", "ASPDM", "ApproxAverage", "hitting"]:
    data = mulit_test(N, 10, method)
    Y = np.mean(data, axis=0)
    std = scipy.stats.sem(data, axis=0)
    conf = std * scipy.stats.t.ppf((1 + .8) / 2., len(Y)-1)
    plt.fill_between(range(len(Y)), Y + conf, Y - conf, color=color_dict[method], alpha=0.25)
    plt.plot(Y, color=color_dict[method], label=method)
plt.legend()
plt.show()
