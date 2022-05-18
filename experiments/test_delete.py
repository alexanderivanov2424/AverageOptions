

import numpy as np
import networkx as nx
from scipy.sparse.linalg import eigsh
import time

#Gnx = nx.path_graph(90)
Gnx = nx.random_regular_graph(d=3, n=1000)

G = nx.to_networkx_graph(Gnx)
A = nx.linalg.laplacian_matrix(G).astype(float)

A = A.todense().astype(float)

k = 5

t = time.time()
evalues, evectors = eigsh(A, int(k / 2) + 1, which='SA')
print(time.time() - t)

t = time.time()
evalues, evectors = np.linalg.eigh(A)
print(time.time() - t)