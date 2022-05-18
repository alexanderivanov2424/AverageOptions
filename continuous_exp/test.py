from operator import mod
from webbrowser import get
import gym
import d4rl 
import numpy as np
import random

from continuous_exp.search import *
from continuous_exp.ballOptions import *
from continuous_exp.utils import *
from continuous_exp.run_cont_vi_exp import *

from continuous_exp.continuousExperiment import Experiment

from options.OptionGeneration import GetOptions

import matplotlib.pyplot as plt
from continuous_exp.dataCollection import loadData, buildAdjacencyFromStates, visualizeStates
from continuous_exp.dataCollection import *

import scipy.spatial.distance

from options.ContApproxAverageOptions import ContApproxAverageOptions

goal = np.array((8, 0))

# env = gym.make('antmaze-umaze-v2')


# env.reset()
# while True:
#     a = sampleAction(env, goal)
#     env.step(a)
#     env.render(mode = "human")




# env.reset()
# traj_len = 50
# samples = 20

# while True:
#     actions = sample_trajectory(env, goal, traj_len, samples)
#     for a in actions:
#         env.step(a)
#         #env.render(mode = "human")
#     d = np.sum(np.square(getPos(env) - goal)) 
#     print(d)
#     if d < 5:
#         traj_len = 10
#         samples = 200
#     if d < .25:
#         break

# while True:
#     env.render(mode = "human")


"CartPole-v0"
"CartPole-v1"
"MountainCar-v0"
"MountainCarContinuous-v0"
"Pendulum-v1"
"Acrobot-v1"

"LunarLander-v2"
"LunarLanderContinuous-v2"
"BipedalWalker-v3"
"BipedalWalkerHardcore-v3"
"CarRacing-v0"

"Blackjack-v1"
"FrozenLake-v1"
"FrozenLake8x8-v1"
"CliffWalking-v0"
"Taxi-v3"

"Reacher-v2"
"Pusher-v2"
"Thrower-v2"
"Striker-v2"
"InvertedPendulum-v2"
"InvertedDoublePendulum-v2"
"HalfCheetah-v2"
"HalfCheetah-v3"
"Hopper-v2"
"Hopper-v3"
"Swimmer-v2"
"Swimmer-v3"
"Walker2d-v2"
"Walker2d-v3"
"Ant-v2"

# env = gym.make('antmaze-umaze-v2')
# env = gym.make('FetchReach-v1')
# env = gym.make('Walker2d-v3')
#env = gym.make('HandReach-v0')

# env.reset()


# min_arr = np.ones((3,)) * 10000
# max_arr = np.ones((3,)) * -10000

# for root in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)]:
#     setEnvVecIdx(env, root, [6,7])
#     for _ in range(1000):
#         a = env.action_space.sample()
#         env.step(a)
#         pos = getEnvPos(env)
#         min_arr = np.minimum(min_arr, pos)
#         max_arr = np.maximum(max_arr, pos)

#     print(min_arr, max_arr)




# exit()

ENV_NAME = "FetchReach-v1"
IDX = [6,7]
roots = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)]

env = gym.make(ENV_NAME)
env.reset()

# states = sampleStatesFromRoots(env, roots, IDX, n_samples=1000)
# saveData(ENV_NAME, f'sampled_states', states)

# states = loadData(ENV_NAME, 'sampled_states')
# states=states[::300]


# A = buildAdjacencyFromStates(env, states, 0, 5, use_pos=True)



# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')

# for i in range(len(states) - 1):
#     for j in range(i+1, len(states)): 
#         if A[i,j]:
#             setEnvVec(env, states[i])
#             p1 = getEnvPos(env)
#             setEnvVec(env, states[j])
#             p2 = getEnvPos(env)
#             ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]])#, 'b-', markeredgewidth=1, zorder=1)


# X, Y, Z = [], [], []
# for s in states:
#     setEnvVec(env, s)
#     pos = getEnvPos(env)
#     X.append(pos[0])
#     Y.append(pos[1])
#     Z.append(pos[2])


# ax.scatter(X, Y, Z, s=1)
# plt.show()

# exit()

# for r in [.1, .2, .3]:
#     discretizeSavedStates(ENV_NAME, f'sampled_states', f'disc_r{r}_states_pos', r=r, env=env)

# exit()

for r in [.1, .2, .3]:
    states = loadData(ENV_NAME, f'disc_r{r}_states_pos')
    # states_full = states = loadData(ENV_NAME, 'sampled_states')
    # for s in states_full:
    #     d = np.sqrt(np.sum(np.square(s[IDX] - states[0][IDX])))
    #     print(d)

    print(len(states))
    A = buildAdjacencyFromStates(env, states, r+.05, 2, use_pos=True)
    #visualizeStates(ENV_NAME, states, IDX, A)#, save_name=f"graph_discR_{r}_eps_{.1}_knn_{5}")
    

    
    import networkx as nx
    G = nx.from_numpy_array(A)
    #nx.draw(G, node_size=5)
    print(nx.is_connected(G))
    #plt.show()
    #exit()

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    for method in ['eigen', 'fiedler', 'ApproxAverage']:
        print("[+]", method)
        options = buildOptions(states, A, 4, method, .1)
        for op in options:
            setEnvVec(env, op.init_vec)
            p1 = getEnvPos(env)
            setEnvVec(env, op.term_vec)
            p2 = getEnvPos(env)
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color=color_dict[method])#, 'b-', markeredgewidth=1, zorder=1),

    plt.show()

    #visualizeStates(ENV_NAME, states, IDX, A)#, save_name=f"graph_discR_{r}_eps_{.1}_knn_{5}")


"""
idx 6 : -1.5, 1
idx 7 : -1.5, 1
idx 8 : -3, 3
idx 9 : -2, 2.5
"""

# for r in [6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10]:
#     discretizeSavedStates(ENV_NAME, f'sampled_states', f'disc_r{r}_states', r=r)

# for r in [6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10]:
#     states = loadData(ENV_NAME, f'disc_r{r}_states')
#     print(len(states))
#     A = buildAdjacencyFromStates(env, states, .5, 10, idx=IDX)
#     visualizeStates(ENV_NAME, states, A, save_name=f"graph_discR_{r}_eps_{.5}_knn_{10}")


