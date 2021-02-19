from simple_rl.tasks import GridWorldMDP, TaxiOOMDP, HanoiMDP#, GymMDP
from simple_rl.tasks.grid_world.GridWorldMDPClass import make_grid_world_from_file


# options
from options.option_generation.fiedler_options import FiedlerOptions
from options.option_generation.eigenoptions import Eigenoptions
from options.option_generation.betweenness_options import BetweennessOptions
from options.option_generation.ASPDM_options import AverageShortestOptions
from new_experiments.ApproxAverageOptions import ApproxAverageOptions

from options.option_generation.graph_drawing_options import GraphDrawingOptions
from options.graph.cover_time import ComputeCoverTime
from options.graph.spectrum import ComputeConnectivity
from options.option_generation.util import GetAdjacencyMatrix, GetIncidenceMatrix

from simple_rl.agents import QLearningAgent, LinearQAgent, RandomAgent#, DQNAgent
from new_experiments.run_experiments import run_agents_on_mdp
from simple_rl.abstraction import AbstractionWrapper, aa_helpers, ActionAbstraction, OnlineAbstractionWrapper

import matplotlib
import matplotlib.path as mpath
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

import numpy as np
import scipy.stats
import gc
import csv
import os

import networkx as nx


matplotlib.style.use('default')

def build_online_subgoal_option_agent(mdp, agent=QLearningAgent, n_ops=4, freqs=100, episodes = 10, steps = 10, method='eigen', name='-online-op'):
    goal_based_aa = ActionAbstraction(prim_actions=mdp.get_actions(), use_prims=True)

    option_agent = OnlineAbstractionWrapper(agent, agent_params={"actions":mdp.get_actions()},
        action_abstr=goal_based_aa, name_ext=name, n_ops=n_ops, freqs=freqs, op_n_episodes=episodes, op_n_steps=steps, max_ops=32, method=method, mdp=mdp)

    return option_agent

def save_exp(exp, exp_name, path="saves//onlineQLearning"):
    for agent in exp.rewards.keys():
        folder = path + "//" + exp_name + "//"
        if not os.path.exists(folder):
            os.makedirs(folder)
        with open(folder + agent.name + ".csv",'a') as f:
            writer = csv.writer(f)
            for instance in exp.rewards[agent].keys():
                episode_rewards = []
                for episode in exp.rewards[agent][instance].keys():
                    episode_rewards.append(exp.rewards[agent][instance][episode])
                writer.writerow(episode_rewards)

def load_episode_rewards(agent, exp_name, path="saves//onlineQLearning"):
    instance_reward_lists =[]
    with open(path + "//" + exp_name + "//" + agent.name + ".csv") as f:
        reader = csv.reader(f)
        for row in reader:
            instance_reward_lists.append(np.array(row, dtype=float))
    return instance_reward_lists


def make_plot(dom, task="", rand_init_and_goal=False, save_and_load=False, n_options=3, n_instances=10, n_episodes=100, n_steps=500, freqs = 100):
    if dom == 'grid':
        mdp = make_grid_world_from_file('options/tasks/' + task + '.txt', rand_init_and_goal=rand_init_and_goal)
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
    else:
        print('Unknown task name: ', task)
        assert(False)

    origMatrix, intToS = GetAdjacencyMatrix(mdp)

    ql_agent = QLearningAgent(actions=mdp.get_actions())

    rand_agent = RandomAgent(mdp.get_actions())

    # TODO: Add an arugment for selecting option generation method.
    fiedler_agent = build_online_subgoal_option_agent(mdp, agent=QLearningAgent, n_ops=n_options, freqs=freqs, episodes=n_episodes, steps=n_steps, method='fiedler', name='-fiedler')
    eigen_agent = build_online_subgoal_option_agent(mdp, agent=QLearningAgent, n_ops=n_options, freqs=freqs, episodes=n_episodes, steps=n_steps, method='eigen', name='-eigen')
    # ASPDM_agent = build_online_subgoal_option_agent(mdp, agent=QLearningAgent, n_ops=n_options, freqs=freqs, episodes=n_episodes, steps=n_steps, method='ASPDM', name='-ASPDM')
    ApproxAverage_agent = build_online_subgoal_option_agent(mdp, agent=QLearningAgent, n_ops=n_options, freqs=freqs, episodes=n_episodes, steps=n_steps, method='ApproxAverage', name='-ApproxAverage')
    # bet_agent = build_online_subgoal_option_agent(mdp, agent=QLearningAgent, n_ops=n_options, freqs=freqs, episodes=n_episodes, steps=n_steps, method='bet', name='-bet')

    exp = run_agents_on_mdp([fiedler_agent, eigen_agent, ApproxAverage_agent, ql_agent, rand_agent], mdp,
        instances=n_instances, episodes=n_episodes, steps=n_steps, track_reward='sum', track_disc_reward=True, reset_at_terminal=False)


    if save_and_load:
        save_exp(exp, dom + task)

    for agent in exp.rewards.keys():

        instance_reward_lists = []
        if save_and_load:
            instance_reward_lists = load_episode_rewards(agent, dom + task)
            instance_reward_lists = np.cumsum(instance_reward_lists, axis=1)
        else:
            for instance in exp.rewards[agent].keys():
                episode_rewards = []
                for episode in exp.rewards[agent][instance].keys():
                    episode_rewards.append(exp.rewards[agent][instance][episode])
                instance_reward_lists.append(np.cumsum(episode_rewards))

        #averaged over instances
        data = np.array(instance_reward_lists)
        Y = np.mean(data, axis=0)
        std = scipy.stats.sem(data, axis=0)
        conf = std * scipy.stats.t.ppf((1 + .95) / 2., len(Y)-1)
        plt.fill_between(range(len(Y)), Y + conf, Y - conf, alpha=0.25)
        plt.plot(Y, label=agent.name)

    plt.title(dom + "  " + task)
    plt.xlabel('episode')
    plt.ylabel('cumulative reward')
    plt.legend()
    plt.show(block=True)
    gc.collect()


SAVE = False
n_options = 4
n_instances = 10
episodes = 100
make_plot("grid", task="9x9grid", save_and_load=SAVE, n_options=n_options, n_instances=n_instances, n_episodes=episodes, n_steps=100, freqs=10000)
make_plot("grid", task="fourroom", save_and_load=SAVE, n_options=n_options, n_instances=n_instances, n_episodes=episodes, n_steps=500, freqs=10000)
make_plot("hanoi", save_and_load=SAVE, n_options=n_options, n_instances=n_instances, n_episodes=episodes, n_steps=500, freqs=10000)
make_plot("taxi", save_and_load=SAVE, n_options=n_options, n_instances=n_instances, n_episodes=episodes, n_steps=500, freqs=10000)
# make_plot("grid", task="Parr", rand_init_and_goal=True, save_and_load=True, n_options=n_options, n_instances=n_instances, n_episodes=100, n_steps=500, freqs=10000)
# make_plot("grid", task="Track2", rand_init_and_goal=True, save_and_load=False, n_options=n_options, n_instances=n_instances, n_episodes=episodes, n_steps=500, freqs=10000)
