from simple_rl.tasks import GridWorldMDP, TaxiOOMDP, HanoiMDP#, GymMDP
from simple_rl.tasks.grid_world.GridWorldMDPClass import make_grid_world_from_file
from simple_rl.tasks.race_track.RaceTrackMDPClass import make_race_track_from_file

# options
from options.FiedlerOptions import FiedlerOptions
from options.EigenOptions import Eigenoptions
from options.AverageOptions import AverageShortestOptions
from options.ApproxAverageOptions import ApproxAverageOptions
from options.HittingOptions import HittingTimeOptions

from options.graph.mdp import GetAdjacencyMatrix

from simple_rl.agents import QLearningAgent, LinearQAgent, RandomAgent#, DQNAgent
from experiments.run_offline_experiments import train_agents_on_mdp#, QLearn_mdp_offline

# from options.OptionClasses.PointOptionMDPWrapperClass import PointOptionMDP
from options.OptionClasses.Option import Option, getGraphFromMDP, constructOptionObject, constructPointOptionObject
from options.OptionClasses.OptionAgent import OptionAgent
from options.OptionGeneration import GetOptions

import matplotlib
import matplotlib.path as mpath
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d


import numpy as np
import scipy.stats
import gc
import random

import networkx as nx

matplotlib.style.use('default')


def make_option_agent(mdp, nx_graph, Matrix, intToS, method='eigen'):
    A = Matrix.copy()

    def call_back(num_options, experiences=None):
        if experiences != None:
            return []
        _, option_i_pairs, _ = GetOptions(A, num_options, method)
        options = [constructPointOptionObject(option_i_pair, nx_graph, intToS) for option_i_pair in option_i_pairs]
        return options

    return OptionAgent(method + "-options", mdp.actions, call_back, default_q=1.0)


def make_plot(dom, task="", rand_init_and_goal=True, n_options=8, n_instances=10, n_episodes=100, n_steps=500):
    if dom == 'grid':
        mdp = make_grid_world_from_file('tasks/' + task + '.txt', rand_init_and_goal=rand_init_and_goal, step_cost=0.0)
    elif dom == 'taxi':
        width = 4
        height = 4
        agent = {"x": 1, "y":1, "has_passenger":0}
        passengers = [{"x":3, "y":2, "dest_x":2, "dest_y": 3, "in_taxi":0}]
        mdp = TaxiOOMDP(width, height, agent, walls=[], passengers=passengers, step_cost=0.0)
    elif dom == 'gym':
        mdp = GymMDP(env_name=task, render=False)
    elif dom == 'hanoi':
        mdp = HanoiMDP(num_pegs=3, num_discs=4, rand_init_goal=rand_init_and_goal, step_cost=0.0)
    elif dom == 'track':
        mdp = make_race_track_from_file('tasks/' + task + '.txt', rand_init_and_goal=rand_init_and_goal, step_cost=0.0)
    else:
        print('Unknown task name: ', task)
        assert(False)

    nx_graph, A, intToS = getGraphFromMDP(mdp)

    eigenAgent = make_option_agent(mdp, nx_graph, A, intToS, "eigen")
    # fiedlerAgent = make_option_agent(mdp, nx_graph, A, intToS, "fiedler")
    # ASPDMAgent = make_option_agent(mdp, nx_graph, A, intToS, "ASPDM")
    ApproxAverageAgent = make_option_agent(mdp, nx_graph, A, intToS, "ApproxAverage")
    # HittingAgent = make_option_agent(mdp, nx_graph, A, intToS, "hitting")
    ql_agent = QLearningAgent(actions=mdp.get_actions(), default_q=1.0)

    # agents = [eigenAgent, fiedlerAgent, ASPDMAgent, ApproxAverageAgent, HittingAgent, ql_agent]
    # agents = [eigenAgent, ASPDMAgent, ApproxAverageAgent, ql_agent]
    agents = [eigenAgent, ApproxAverageAgent, ql_agent]

    for agent in agents:
        if agent == ql_agent:
            continue
        agent.generate_options(n_options)

    experiment = train_agents_on_mdp(agents, mdp, instances=n_instances, episodes=n_episodes, steps=n_steps, episode_sample_rate = 1)

    # for s in eigenAgent.q_func:
    #     print(s, end=" ")
    #     max_a = None
    #     for a in eigenAgent.q_func[s]:
    #         if max_a == None or eigenAgent.q_func[s][a] > eigenAgent.q_func[s][max_a]:
    #             max_a = a
    #             continue
    #     print(max_a)

    color_dict = {  "eigen-options":"tab:blue",
                    "fiedler-options":"tab:orange",
                    "ASPDM-options":"tab:green",
                    "ApproxAverage-options":"tab:red",
                    "hitting-options":"cyan",
                    "Q-learning":"black",
                    "Random":"tab:purple",
                    }

    fig, ax = plt.subplots()

    for agent_name in experiment.keys():
        data = np.array(experiment[agent_name])
        data = np.apply_along_axis(lambda x: np.convolve(x, np.ones(5)/5, "valid"), 1, data)
        Y = np.mean(data, axis=0)
        se = scipy.stats.sem(data, axis=0)
        conf = se * scipy.stats.t.ppf((1 + .8) / 2., len(Y)-1)
        plt.fill_between(range(len(Y)), Y + conf, Y - conf, color=color_dict[agent_name], alpha=0.25)
        plt.plot(range(len(Y)), Y, color=color_dict[agent_name], label=agent_name)

    plt.title(dom + "  " + task)
    plt.xlabel('episode')
    plt.ylabel('reward')
    plt.legend()
    exp_name = task if dom == 'grid' else dom
    plt.savefig(f'Plots/offline_test_{exp_name}_inst{n_instances}_ep{n_episodes}_op{n_options}.png')
    # plt.show(block=True)
    gc.collect()
    plt.cla()


np.random.seed(0)
random.seed(0)

RAND_INIT = True
n_options = 8#8
n_instances = 200 #200
episodes = 100 #100
make_plot("grid", task="9x9grid", rand_init_and_goal=RAND_INIT, n_options=n_options, n_instances=n_instances, n_episodes=episodes, n_steps=100)
make_plot("grid", task="fourroom", rand_init_and_goal=RAND_INIT, n_options=n_options, n_instances=n_instances, n_episodes=episodes, n_steps=100)
make_plot("grid", task="tworoom", rand_init_and_goal=RAND_INIT, n_options=n_options, n_instances=n_instances, n_episodes=episodes, n_steps=100)
make_plot("grid", task="twohall", rand_init_and_goal=RAND_INIT, n_options=n_options, n_instances=n_instances, n_episodes=episodes, n_steps=100)
make_plot("hanoi", rand_init_and_goal=RAND_INIT, n_options=n_options, n_instances=n_instances, n_episodes=episodes, n_steps=500)
# make_plot("taxi", rand_init_and_goal=RAND_INIT, n_options=n_options, n_instances=n_instances, n_episodes=episodes, n_steps=500)
make_plot("grid", task="ParrUp", rand_init_and_goal=RAND_INIT, n_options=n_options, n_instances=n_instances, n_episodes=episodes, n_steps=500)
make_plot("grid", task="Track2", rand_init_and_goal=RAND_INIT, n_options=n_options, n_instances=n_instances, n_episodes=episodes, n_steps=500)
# make_plot("grid", task="Parr", rand_init_and_goal=RAND_INIT, n_options=n_options, n_instances=n_instances, n_episodes=episodes, n_steps=500)
