import random
import numpy as np
import networkx as nx
from options.graph.spectrum import ComputeFiedlerVector, ComputeConnectivity



def onehot(length, i):
    assert(type(i) is int)
    ret = np.zeros(length, dtype=int)
    ret[i] = 1
    return ret


def neighbor(graph, n):
    assert(graph.ndim == 2)
    array = np.array(graph)[n]
    # print('array=', array)
    l = []
    for i in range(len(array)):

        if array[i] == 1:
            l.append(i)
    return l


def AddEdge(G, vi, vj):
    augGraph = G.copy()
    # print('augGraph', augGraph)
    augGraph[vi, vj] = 1
    augGraph[vj, vi] = 1
    return augGraph


def GetRandomWalk(G):
    ##########################
    # PLEASE WRITE A TEST CODE
    ##########################
    """
    Given an adjacency matrix, return a random walk matrix where the sum of each row is normalized to 1
    """
    P = (G.T / G.sum(axis=1)).T
    return P

def GetRadius(D, C):
    nV = D.shape[0]

    maxd = -1
    for i in range(nV):
        mind = 10000
        for c in range(nV):
            if C[c] == 1:
                dic = D[c][i]
                if dic < mind:
                    mind = dic
        if mind > maxd:
            maxd = mind

    return maxd

def DeriveGraph(D, R):
    """
    Return Gr = (V, Er) where Er = {(u, r) : d(u, v) <= R}
    """
    Gbool = D <= R
    G = Gbool.astype(int)
    G = G - np.identity(D.shape[0])
    # print("G = ", G)
    return G

def GetCost(G):
    # TODO: Implement GetCost
    # Given an adjacency matrix, return all-pair shortest path distance
    D = np.full_like(G, -1, dtype=int)
    N = int(G.shape[0])

    mt = G
    distance = 1
    while distance < N:
        for x in range(N):
            for y in range(N):
                if D[x][y] == -1 and mt[x][y]:
                    D[x][y] = distance
        mt = np.matmul(mt, G)
        distance += 1

    for x in range(N):
        D[x][x] = 0
    return D



# def HittingTime(G, t):
#     """
#     Given a graph adjacency matrix and a start and goal state,
#     return a hitting time.
#     """
#     A = G.copy()
#     A[t, :] = 0
#     A[t, t] = 1
#
#     # print('A', A)
#     A = (A.T / A.sum(axis=1)).T
#     # print('A', A)
#     B = A.copy()
#     Z = []
#     for n in range(G.shape[0] * G.shape[0] * 2):
#         Z.append(B[:, t]) # TODO: We can get the whole vector B[:, t] to speedup by n times
#         B = np.dot(B, A)
#
#     ret = np.zeros_like(Z[0])
#     for n in range(len(Z)):
#         if n == 0:
#             ret += Z[n] * (n+1)
#         else:
#             ret += (Z[n] - Z[n-1]) * (n+1)
#     if any(Z[len(Z) - 1] < 1):
#         ret += (1 - Z[len(Z)-1]) * (len(Z))
#     # print('Z', Z)
#     # print('ret', ret)
#     return ret

def ComputeCoverTimeS(G, s, sample=1000):
    ##########################
    # PLEASE WRITE A TEST CODE
    ##########################
    '''
    Args:
        G (numpy 2d array): Adjacency matrix (may be an incidence matrix).
        s (integer): index of the initial state
        sample (integer): number of trajectories to sample
    Returns:
        (float): the expected cover time from state s
    Summary:
        Given a graph adjacency matrix, return the expected cover time starting from node s. We sample a set of trajectories to get it.
    '''

    N = G.shape[0]

    n_steps = []

    for i in range(sample):
        visited = np.zeros(N, dtype=int)
        visited[s] = 1
        cur_s = s
        cur_steps = 0

        while any(visited == 0):
            s_neighbor = neighbor(G, cur_s)
            next_s = random.choice(s_neighbor)
            visited[next_s] = 1
            cur_s = next_s
            cur_steps += 1

        n_steps.append(cur_steps)

    # print('n_steps=', n_steps)

    avg_steps = sum(n_steps) / sample
    return avg_steps

def ComputeCoverTime(G, samples=1000):
    ##########################
    # PLEASE WRITE A TEST CODE
    ##########################
    '''
    Args:
        G (numpy 2d array): Adjacency matrix (or incidence matrix)
    Returns:
        (float): the expected cover time
    Summary:
        Given a graph adjacency matrix, return the expected cover time.
    '''
    N = G.shape[0]

    c_sum = 0

    for i in range(samples):
        init = random.randint(0, N-1)
        c_i = ComputeCoverTimeS(G, init, sample=1)
        c_sum += c_i

    return float(c_sum) / float(samples)
