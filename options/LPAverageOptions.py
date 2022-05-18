from ast import Pass
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
def JainVaziraniOptions(G,k):
    A = G.copy()
    graph = nx.to_networkx_graph(A)

    D_dict = nx.all_pairs_shortest_path_length(nx.to_networkx_graph(A))
    D = np.zeros(A.shape,dtype='int')
    for source in D_dict:
        for target in source[1].keys():
            D[source[0],target] = source[1][target]

    N = len(A)

    np.set_printoptions(linewidth=100000)

    P = np.zeros(N*N*2)
    C = np.zeros((N,N*N*2))

    for i in range(N):
        for j in range(N):
            P[N*i + 2*j] = max(0, D[i,j] - 2)
            P[N*i + 2*j + 1] = max(0, D[j,i] - 2)
            for i_ in range(N):
                C[i_, N*i + 2*j] = D[i_,i]
                C[i_, N*i + 2*j + 1] = D[i_,i]

    #S = kFLWP(k+1, P, C)

    S,_ = FLWP(np.array([1 for _ in range(N)]), P, C)
    print(S)
    exit()

    options = pack_options(S, A)
    return A, options


def smallest_positive(A):
    return np.where(A >= 0, A, np.inf).min()

"""
k Facility Location with Penalties
k - number of facilities to be opened
P - (m,) penalties for each city
C - (n,m) distance from facility to city

n - num facilities, m - num cities

return
S - list of facilities to be opened.
"""
def kFLWP(k, P, C):
    N,M = C.shape

    z1 = 0
    S1 = None
    B1 = None

    z2 = 1
    S2 = None
    B2 = None

    z = 1

    import time

    t = time.time()

    while z2 - z1 > .0000001:

        F = np.array([z for _ in range(N)])

        t = time.time()
        S,B = FLWP(F,P,C)
        print(time.time() - t)

        m = len(S)
        print(f"Searching with z:{z}", f"found {m} when want {k}")
        if m < k:
            if z == z2:
                z, z2 = z*2, z2*2
            z2, S2, B2 = z, S, B
            z = (z1 + z2)/2
        elif m > k:
            z1, S1, B1 = z, S, B
            z = (z1 + z2)/2
        else:
            return S
    
    if S1 is None:
        if len(S) < 2:
            S = list(S)
            while len(S) < 2:
                for i in range(N):
                    if not i in S:
                        S.append(i)
                        break
            return S
        return S

    S = list([i for i in S1])

    for i in S2:
        if len(S) == k:
            break
        if not i in S1:
            can_be_added = True
            for i_ in S1:
                if not can_be_added:
                    break
                for j in range(M):
                    if B1[i,j] > 0 and B2[i,j] > 0 and B1[i_,j] > 0 and B2[i_,j] > 0:
                        can_be_added = False
                        break
            if can_be_added:
                S.append(i)
    
    return S
            
    

    


"""
Facility Location with Penalties
F - (n,) facility opening costs
P - (m,) penalties for each city
C - (n,m) distance from facility to city

n - num facilities, m - num cities

return
S - list of facilities to be opened.
b - (n,n) matrix of 'b' values  b = max(a-C, 0)
"""
def FLWP(F, P, C):
    N, M = C.shape

    t = 0
    a = np.zeros(M) # a values start at 0 and grow with t
    # b values are b_ij = a_j - C_ij when a_j > C_ij

    L = np.ones(M) # all cities start unlocked
    U = np.zeros(N) # all facilities start closed
    O = np.ones(M) # no cities start as outliers

    W = {} # witness of each city | city index -> witness facility index
    T = {} # time when each facility is payed out | facility index -> time

    BIG_C = np.max(C) + 1
    BIG_P = np.max(P) + 1
    BIG_F = np.max(F) + 1

    # phase 1

    import time

    while len(W.keys()) < M: #run until all cities have a witness
        # pay facility time
        t_f = (F - np.sum(np.where(a-C > 0, a-C, 0),axis=1) - BIG_F*U) / np.sum(a-C >= 0,axis=1) #when row of b is zero result is inf
        # link city to payed facility time
        t_l = ((C-a).T + BIG_C * (U-1)).T + BIG_C*(L * O - 1)
        # make city outlier due to penalty time
        t_p = P - a + BIG_P*(L * O - 1)

        times = [smallest_positive(t_f), smallest_positive(t_l), smallest_positive(t_p)]
        case = np.argmin(times)
        t_min = np.min(times)
        t += t_min

        #print(str(case) + " ", end="")
        print(times)
        if(t_min == np.inf):
            break


        if case == 0:
            # open facility
            facility = np.argwhere(t_f == smallest_positive(t_f))[0][0]
            U[facility] = 1 # open facility
            T[facility] = t # save time when payed for
            for city in range(N):
                if C[facility, city] < a[city]:
                    L[city] = 0 # lock city
                    W[city] = facility # set cityy witness
        elif case == 1:
            # link city to facility
            facility, city = np.argwhere(t_l == smallest_positive(t_l))[0]
            L[city] = 0 # lock city
            W[city] = facility # set cityy witness
        elif case == 2:
            # lock city at penalty
            city = np.argwhere(t_p == smallest_positive(t_p))[0][0]
            O[city] = 0 # make city outlier
        
        # increase 'a' values
        a += t_min * (L * O)

    print()
    print(T)

    b = a-C >= 0

    while len(T) > 0:
        earliest_facility = list(T.keys())[0]
        for facility, t in T.items():
            if T[earliest_facility] > t:
                earliest_facility = facility
        

        for conn_city in np.argwhere(b[earliest_facility]==1).flatten():
            for conn_facility in np.argwhere(b[:,conn_city]==1).flatten():
                if conn_facility == earliest_facility:
                    continue
                U[conn_facility] = 0 # delete facility
                T.pop(conn_facility, None) #remove from time dict

                for c, f in W.items(): # change any witnesses to deleted facility
                    if f == conn_facility and O[c] == 1:
                        W[c] = earliest_facility

        T.pop(earliest_facility, None)


    return np.argwhere(U == 1).flatten(), (a-C) > 0



def test():
    N = 10
    #Gnx = nx.cycle_graph(N)
    Gnx = nx.path_graph(N)

    #Gnx = nx.random_regular_graph(d=2, n=N)
    #Gnx = nx.barabasi_albert_graph(n=N,m=1)
    A = nx.to_numpy_matrix(Gnx).astype(dtype='int')

    A_, options = JainVaziraniOptions(A, 1)
    print(options)

if __name__ == "__main__":
    test()