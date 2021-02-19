from simple_rl.tasks import GridWorldMDP, TaxiOOMDP, HanoiMDP#, GymMDP
from simple_rl.tasks.grid_world.GridWorldMDPClass import make_grid_world_from_file
from simple_rl.tasks.race_track.RaceTrackMDPClass import make_race_track_from_file

# options
from options.FiedlerOptions import FiedlerOptions
from options.EigenOptions import Eigenoptions
from options.AverageOptions import AverageShortestOptions
from options.ApproxAverageOptions import ApproxAverageOptions

from options.graph.mdp import GetAdjacencyMatrix

from simple_rl.agents import QLearningAgent, LinearQAgent, RandomAgent#, DQNAgent
from experiments.run_offline_experiments import QLearn_mdp_offline

from options.OptionClasses.PointOptionMDPWrapperClass import PointOptionMDP
from options.OptionClasses.Option import Option

import matplotlib
import matplotlib.path as mpath
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

import numpy as np
import scipy.stats
import gc

import networkx as nx

matplotlib.style.use('default')


def GetOption(mdp, k=1, sample=False, matrix=None, intToS=None, method='eigen'):
    if matrix is None:
        A, intToS = GetAdjacencyMatrix(mdp)
    else:
        A = matrix

    print("Getting options by method: " + method)

    if method == 'eigen':
        B, options, vectors = Eigenoptions(A, k*2)
    elif method == 'fiedler':
        B, options, _, vectors = FiedlerOptions(A, k*2)
        options = [(opp[0][0],opp[1][0]) for opp in options]
    elif method == 'ASPDM':
        B, options = AverageShortestOptions(A, None, k)
        vectors = None
    elif method == 'ApproxAverage':
        B, options = ApproxAverageOptions(A, k)
        vectors = None
    elif method == 'bet':
        # TODO: B is empty.
        B, options, vectors = BetweennessOptions(A, k)

    return B, options, intToS, vectors

def get_option_objects(matrix_A, intToS, options):
    graph = nx.to_networkx_graph(matrix_A)
    # pair wise distances in graph
    D_dict = nx.all_pairs_shortest_path_length(graph)
    D = np.zeros(matrix_A.shape,dtype='int')
    for source in D_dict:
        for target in range(len(matrix_A)):
            D[source[0],target] = source[1][target]

    state_options = []
    for option in options:
        path = nx.shortest_path(graph, source=option[0], target=option[1])
        state_path = [intToS[p] for p in path]
        state_options.append(Option(intToS[option[0]], intToS[option[1]], state_path, D[option[0],option[1]]))

    return state_options

def test_offline_options(mdp, matrix_A, intToS, options, n_instances=10, n_episodes=100, n_steps=500, agent_type='q', name="q-learning"): #q or rand
    if agent_type == 'q':
        agent = QLearningAgent(actions=mdp.get_actions(), default_q=1.0)
    else:
        agent = RandomAgent(mdp.get_actions())

    option_mdp = PointOptionMDP(mdp)
    state_options = get_option_objects(matrix_A, intToS, options)
    option_mdp.add_options(state_options)

    print("Running Experiment with " + name)
    instance_rewards = QLearn_mdp_offline(agent, option_mdp, instances=n_instances, episodes=n_episodes, steps=n_steps)

    instance_reward_lists = []
    for episode_rewards in instance_rewards:
        instance_reward_lists.append(np.cumsum(episode_rewards))

    #averaged over instances
    data = np.array(instance_reward_lists)
    Y = np.mean(data, axis=0)
    std = scipy.stats.sem(data, axis=0)
    conf = std * scipy.stats.t.ppf((1 + .95) / 2., len(Y)-1)
    plt.fill_between(range(len(Y)), Y + conf, Y - conf, alpha=0.25)
    plt.plot(Y, label=name)



def make_plot(dom, task="", rand_init_and_goal=True, n_options=3, n_instances=10, n_episodes=100, n_steps=500):
    if dom == 'grid':
        mdp = make_grid_world_from_file('tasks/' + task + '.txt', rand_init_and_goal=rand_init_and_goal)
    elif dom == 'taxi':
        width = 4
        height = 4
        agent = {"x": 1, "y":1, "has_passenger":0}
        passengers = [{"x":3, "y":2, "dest_x":2, "dest_y": 3, "in_taxi":0}]
        mdp = TaxiOOMDP(width, height, agent, walls=[], passengers=passengers)
    elif dom == 'gym':
        mdp = GymMDP(env_name=task, render=False)
    elif dom == 'hanoi':
        mdp = HanoiMDP(num_pegs=3, num_discs=4)
    elif dom == 'track':
        mdp = make_race_track_from_file('tasks/' + task + '.txt', rand_init_and_goal=rand_init_and_goal)
    else:
        print('Unknown task name: ', task)
        assert(False)

    origMatrix, intToS = GetAdjacencyMatrix(mdp)

    fiedlerMatrix, foptions, _, _ = GetOption(mdp, n_options, matrix=origMatrix, intToS=intToS, method='fiedler')
    eigenMatrix, eoptions, _, _ = GetOption(mdp, n_options, matrix=origMatrix, intToS=intToS, method='eigen')
    ASPDMMatrix, ASPDMoptions, _, _ = GetOption(mdp, n_options, matrix=origMatrix, intToS=intToS, method='ASPDM')
    # ApproxAverageMatrix, ApproxAverageoptions, _, _ = GetOption(mdp, n_options, matrix=origMatrix, intToS=intToS, method='ApproxAverage')


    test_offline_options(mdp, origMatrix, intToS, foptions, n_instances=n_instances, n_episodes=n_episodes, n_steps=n_steps, name="fiedler")
    test_offline_options(mdp, origMatrix, intToS, eoptions, n_instances=n_instances, n_episodes=n_episodes, n_steps=n_steps, name="eigen")
    test_offline_options(mdp, origMatrix, intToS, ASPDMoptions, n_instances=n_instances, n_episodes=n_episodes, n_steps=n_steps, name="average options")
    test_offline_options(mdp, origMatrix, intToS, [], n_instances=n_instances, n_episodes=n_episodes, n_steps=n_steps, name="q-learning")
    test_offline_options(mdp, origMatrix, intToS, [], n_instances=n_instances, n_episodes=n_episodes, n_steps=n_steps, agent_type="rand", name="random")


    plt.title(dom + "  " + task)
    plt.xlabel('episode')
    plt.ylabel('cumulative reward')
    plt.legend()
    plt.show(block=True)
    gc.collect()


RAND_INIT = True
n_options = 8
n_instances = 10
episodes = 500
make_plot("grid", task="9x9grid", rand_init_and_goal=RAND_INIT, n_options=n_options, n_instances=n_instances, n_episodes=episodes, n_steps=50)
# make_plot("grid", task="fourroom", rand_init_and_goal=RAND_INIT, n_options=n_options, n_instances=n_instances, n_episodes=episodes, n_steps=100)
# make_plot("grid", task="tworoom", rand_init_and_goal=RAND_INIT, n_options=n_options, n_instances=n_instances, n_episodes=episodes, n_steps=100)
# make_plot("grid", task="twohall", rand_init_and_goal=RAND_INIT, n_options=n_options, n_instances=n_instances, n_episodes=episodes, n_steps=100)
# make_plot("hanoi", rand_init_and_goal=RAND_INIT, n_options=n_options, n_instances=n_instances, n_episodes=episodes, n_steps=500)
# make_plot("taxi", rand_init_and_goal=RAND_INIT, n_options=n_options, n_instances=n_instances, n_episodes=episodes, n_steps=500)
# make_plot("grid", task="ParrUp", rand_init_and_goal=RAND_INIT, n_options=n_options, n_instances=n_instances, n_episodes=episodes, n_steps=500)
# make_plot("grid", task="Track2", rand_init_and_goal=RAND_INIT, n_options=n_options, n_instances=n_instances, n_episodes=episodes, n_steps=500)
