from fileinput import filename
import gym
import d4rl 
import numpy as np

from continuous_exp.utils import getEnvPos, getEnvVec, setEnvVec, setEnvVecIdx, getVecIdx, vecToPos
from continuous_exp.utils import saveData, loadData, saveFig

from sklearn.neighbors import KDTree

import matplotlib.pyplot as plt

from tqdm import tqdm



def sampleStates(env, n_samples=1000000):
    states = []
    for step in tqdm(range(n_samples)):
        states.append(getEnvVec(env))
        env.step(env.action_space.sample())

    return states


def sampleStatesFromRoots(env, root_points, idx, n_samples=5000):
    states = []
    with tqdm(total=len(root_points) * n_samples) as pbar:
        for root in root_points:
            env.reset()
            setEnvVecIdx(env, root, idx)
            for step in range(n_samples):
                states.append(getEnvVec(env))
                env.step(env.action_space.sample())
                pbar.update(1)

    return states

def discretizePoints(points, radius, idx=None, env=None):
    if not idx is None:
        def process(point):
            return point[idx]
    elif not env is None:
        def process(point):
            setEnvVec(env, point)
            return getEnvPos(env)
    else:
        def process(point):
            return point

    L = len(points)
    discPoints = [process(points[0])]
    discPointsId = [0]
    tree = KDTree(discPoints, leaf_size=2)
    step = 50

    i = 1
    while i < L:
        dist = tree.query_radius([process(p) for p in points[i:i+step]], r=radius)#, return_distance=True, sort_results=True)[0]
        shift = step
        for j,d in enumerate(dist):
            if len(d) == 0:
                discPoints.append(process(points[i+j]))
                discPointsId.append(i+j)
                tree = KDTree(discPoints, leaf_size=2)
                shift = j + 1
                break
        i += shift
        print(len(discPoints), i, i/L, " "*10, end="\r")

    return points[discPointsId]

def discretizeSavedStates(env_name, file_name, save_name, r, idx=None, env=None):
    states = loadData(env_name, file_name)
    points = np.array(states)

    discPoints = discretizePoints(points, r, idx=idx, env=env)

    saveData(env_name, save_name, discPoints)

"""
Returns graph of states where s and s' are connected if
d(s,s') < e
OR
s' is a k-nearest neighbor of s
"""
def buildAdjacencyFromStates(env, states, e, k, use_pos=False, idx=None):
    N = len(states)
    A = np.zeros((N,N), dtype='int')
    
    points = [vecToPos(env, s, use_pos=use_pos, idx=idx) for s in states]

    tree = KDTree(points, metric='euclidean', leaf_size=1)

    for i, p in enumerate(points):
        I_knn = tree.query([p * 1.0000001], k=k+1, return_distance=False)
        if I_knn[0].size > 0 and I_knn[0].size < N-1:
            for j in I_knn[0]:
                A[i,j] = 1
                A[j,i] = 1
        
        I_e = tree.query_radius([p * 1.0000001], e, return_distance=False)
        if I_e[0].size > 0 and I_e[0].size < N-1:
            for j in I_e[0]:
                A[i,j] = 1
                A[j,i] = 1

    return A

def visualizeStates(env_name, states, idx, A=None, save_name=None):
    X = [s[idx[0]] for s in states]
    Y = [s[idx[1]] for s in states]

    plt.cla()

    if not A is None:
        for i in range(len(states) - 1):
            for j in range(i+1, len(states)): 
                if A[i,j]:
                    plt.plot([states[i][idx[0]], states[j][idx[0]]], [states[i][idx[1]], states[j][idx[1]]], 'b-', markeredgewidth=1, zorder=1)

    plt.scatter(X,Y,s=2,color="red",zorder=2)

    if save_name is None:
        plt.show()
    else:
        r = save_name.split("_")[2]
        plt.title(f"Num states: {len(states)} Discretization factor: {r}")
        saveFig(env_name, save_name)

if __name__ == "__main__":

    # ENV_NAME = "antmaze-umaze-v2"
    # IDX = [0,1]
    # roots = [(0,0), (2,0), (4,0), (6,0), (8,0), (8,2), (8,4), (8,6), (8,8), (6,8), (4,8), (2,8), (0,8)]
    # disc_factors = [6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10]
    # eps = .5
    # knn = 10

    ENV_NAME = "FetchReach-v1"
    IDX = [6,7]
    roots = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)]
    disc_factors = [5.5, 6, 6.5]
    eps = .1
    knn = 5

    env = gym.make(ENV_NAME)
    env.reset()

    states = sampleStatesFromRoots(env, roots, IDX, n_samples=10000)
    saveData(ENV_NAME, f'sampled_states', states)

    states = loadData(ENV_NAME, 'sampled_states')
    print(len(states))
    visualizeStates(ENV_NAME, states, IDX)

    for r in disc_factors:
        discretizeSavedStates(ENV_NAME, f'sampled_states', f'disc_r{r}_states', r=r)

    for r in disc_factors:
        states = loadData(ENV_NAME, f'disc_r{r}_states')
        print(len(states))
        A = buildAdjacencyFromStates(env, states, eps, knn, idx=IDX)
        visualizeStates(ENV_NAME, states, A, save_name=f"graph_discR_{r}_eps_{eps}_knn_{knn}")

