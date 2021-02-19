import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

import itertools



"""
A - adjacency matrix
D - distance between pairs
W - weight matrix
P - cost
"""
def Get_A_D_W_P_Matrices(G, Pairs, delta):
    A = G.copy()

    # pair wise distances in graph
    D_dict = nx.all_pairs_shortest_path_length(nx.to_networkx_graph(A))
    D = np.zeros(A.shape,dtype='int')
    for source in D_dict:
        for target in range(len(A)):
            D[source[0],target] = source[1][target]

    # weight of each node pair
    try:
        W = get_weight_matrix(len(A),Pairs)
    except:
        W = np.ones((len(A),len(A)),dtype='int')


    P = np.clip(W*(D-2*delta),0,None)

    return A, D, W, P

def getHittingTimeTarget(B, t, N=10000, epsilon=.01):
    power = np.ones((B.shape[0])) / B.shape[0]
    N_t = np.zeros((1,B.shape[0]))
    B_ = B.copy()
    B_[t,:] = 0 #TODO not exactly as in paper need to fix

    for i in range(1,N):
        power = np.matmul(power, B_)
        N_t += power
        if np.linalg.norm(i*power) < epsilon:
            break
    return N_t

def getHittingTime(G):
    A = G.copy()
    B = (A.T/ np.sum(A, axis=1)).T
    H = np.zeros(A.shape)
    for target in range(A.shape[0]):
        H[:,target] = getHittingTimeTarget(B, target)
    return H

def get_expected_hitting_time_to_set(G, S, N=10000, epsilon=.01):
    A = G.copy()
    B = A / np.sum(A, axis=1)
    power = np.ones((B.shape[0])) / B.shape[0]
    expected_hitting_time = 0
    B[S,:] = 0
    for i in range(1,N):
        power = np.ravel(np.matmul(power, B))
        hits = np.sum(power[S])
        expected_hitting_time += hits * i
        if hits * i < epsilon:
            break
    return expected_hitting_time

"""
A - adjacency matrix
D - distance between pairs
W - weight matrix
P - cost
"""
def Get_A_H_W_P_Matrices(G, delta):
    A = G.copy()

    H = getHittingTime(G)

    # weight of each node pair
    W = np.ones((len(A),len(A)),dtype='int')

    P = np.clip(W*(H-2*delta),0,None)

    return A, H, W, P

"""
S - set of facilities
A - adjacency matrix
"""
def pack_options(S, A):
    options = []
    for i in range(1,len(S)):
        option = (S[0], S[i])
        options.append(option)

        A[option[0],option[1]] = 1
        A[option[1],option[0]] = 1

    return A, options

def HittingTimeOptions(G, k, delta = 1):
    A, H, W, P = Get_A_H_W_P_Matrices(G, delta)

    #S is a list of facilities (indicies)
    def cost(S):
        cost = 0
        distances = np.amin(H[:,S],axis=1)
        for city in range(len(A)):
            cost += np.sum(np.minimum(distances[city], P[city]))
            cost += np.sum(np.minimum(distances[city], P[:,city]))
        return cost

        # return np.sum(np.minimum(np.amin(H[:,S],axis=1).T, P.T).T)
        # return get_expected_hitting_time_to_set(G, S)

    S = np.random.choice(range(0,len(A)), k+1, replace=False) #need k+1 nodes for k edge star
    while True:
        min_cost = cost(S)
        found_better = False
        # loop over all a to remove from S and all b to add
        for a in S:
            for b in range(len(A)):
                if b in S:
                    continue
                S_ = np.copy(S)
                S_smaller = np.delete(S_, np.argwhere(S_==a))
                S_ = np.append(S_smaller, b)

                c = cost(S_)
                if c < min_cost:
                    found_better = True
                    S = S_
                    min_cost = c

                if found_better:
                    break
            else:
                continue
            break

        if not found_better:
            break

    A, options = pack_options(S, A)
    return A, options



if __name__ == "__main__":
    N = 3
    # Gnx = nx.cycle_graph(N)
    Gnx = nx.path_graph(N)
    # Gnx = nx.random_regular_graph(d=2, n=N)
    # Gnx = nx.barabasi_albert_graph(n=N,m=1)

    graph = nx.to_numpy_matrix(Gnx).astype(dtype='int')

    print(get_expected_hitting_time_to_set(graph, [0,1,N-1]))

    # A, options = HittingTimeOptions(graph, 3)
    # print(options)
