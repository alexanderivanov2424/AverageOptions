import numpy as np
import random
from time import sleep

from simple_rl.tasks import GridWorldMDP
# from simple_rl.tasks import GymMDP # Gym

from simple_rl.planning.ValueIterationClass import ValueIteration

def GetAdjacencyMatrix(mdp):
    # print('mdp type=', type(mdp))
    vi = ValueIteration(mdp) # TODO: the VI class does sampling which doesn't work for stochastic planning.
    vi.run_vi()
    A, states = vi.compute_adjacency_matrix()

    for k in range(A.shape[0]):
        A[k][k] = 0

    intToS = dict()
    for i, s in enumerate(states):
        intToS[i] = s
    return A, intToS # (matrix, dict)
