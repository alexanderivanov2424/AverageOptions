

from simple_rl.tasks import GridWorldMDP, TaxiOOMDP, HanoiMDP#, GymMDP
from simple_rl.tasks.grid_world.GridWorldMDPClass import make_grid_world_from_file
from simple_rl.tasks.race_track.RaceTrackMDPClass import make_race_track_from_file

from options.FiedlerOptions import FiedlerOptions
from options.EigenOptions import Eigenoptions
from options.AverageOptions import AverageShortestOptions
from options.ApproxAverageOptions import ApproxAverageOptions
from options.HittingOptions import HittingTimeOptions

from options.graph.mdp import GetAdjacencyMatrix

from options.OptionClasses.Option import getGraphFromMDP
from options.OptionGeneration import GetOptions

import networkx as nx
import numpy as np
import random
import json
import matplotlib
import matplotlib.pyplot as plt

matplotlib.style.use('default')
matplotlib.use('TkAgg')


use_ASPDM = True

def type_to_name(type):
    if type == "eigen":
        return "Eigen Options"
    if type == "fiedler":
        return "Covering Options"
    if type == "ASPDM":
        return "Average Options"
    if type == "ApproxAverage":
        return "Fast Average Options"

def task_to_title(dom, task):
    if dom == "grid":
        if "9x9grid" in task:
            return "9x9 grid"
        if "18x18grid"  in task:
            return "18x18 grid"
        if "fourroom"  in task:
            return "fourroom"
        if "tworoom"  in task:
            return "tworoom"
    return dom

"""
Run goal conditioned Value Itteration on given adjacency matrix for all possible goal states
return number of itterations before convergence.
"""
def run_value_itteration(A, gamma=.99, epsilon=0):
    N = A.shape[0]

    P_orig = A / np.sum(A, axis=1)

    
    iter_per_goal = []

    for g in range(N):
        sum_iter = 0
        V = np.zeros((N, 1))

        P = np.copy(P_orig)
        P[g, :] = 0

        for i in range(1000):
            V_ = np.max(np.multiply(P, (V + np.reshape(np.eye(N)[g], V.shape)).T), axis=1)
            
            if np.max(np.abs(V - V_)) <= epsilon:
                sum_iter += i
                break
            V = V_
        iter_per_goal.append(sum_iter)
    return np.sum(iter_per_goal) / N, np.std(iter_per_goal)/(N**.5)

def itterations_over_options_exp(experiment, method, A, num_options_list):
    experiment[method] = {"num_ops": [], "itter": [], "conf":[]}

    for num_ops in num_options_list:
        B, option_i_pairs, _ = GetOptions(A, num_ops, method)
        print(method, option_i_pairs)
        avg_itter, conf = run_value_itteration(B)
        experiment[method]["num_ops"].append(num_ops)
        experiment[method]["itter"].append(avg_itter)
        experiment[method]["conf"].append(conf)
    


def run_exp(exp_name, num_options_list, dom, task="", plot = True):
    np.random.seed(0)
    random.seed(0)
    print("Running:", dom, task)

    if dom == 'grid':
        mdp = make_grid_world_from_file('tasks/' + task + '.txt', step_cost=0.0)
    elif dom == 'taxi-S':
        width = 4
        height = 4
        agent = {"x": 1, "y":1, "has_passenger":0}
        passengers = [{"x":3, "y":2, "dest_x":2, "dest_y": 3, "in_taxi":0}]
        mdp = TaxiOOMDP(width, height, agent, walls=[], passengers=passengers, step_cost=0.0)
    elif dom == 'taxi':
        width = 5
        height = 5
        agent = {"x": 1, "y":1, "has_passenger":0}
        passengers = [{"x":3, "y":1, "dest_x":1, "dest_y": 3, "in_taxi":0}]
        mdp = TaxiOOMDP(width, height, agent, walls=[], passengers=passengers, step_cost=0.0)
    elif dom == 'taxi-L':
        width = 7
        height = 7
        agent = {"x": 1, "y":1, "has_passenger":0}
        passengers = [{"x":4, "y":1, "dest_x":1, "dest_y": 4, "in_taxi":0}]
        mdp = TaxiOOMDP(width, height, agent, walls=[], passengers=passengers, step_cost=0.0)
    # elif dom == 'gym':
    #     mdp = GymMDP(env_name=task, render=False)
    elif dom == 'hanoi':
        mdp = HanoiMDP(num_pegs=3, num_discs=4, step_cost=0.0)
    elif dom == 'hanoi-L':
        mdp = HanoiMDP(num_pegs=3, num_discs=6,  step_cost=0.0)
    elif dom == 'track':
        mdp = make_race_track_from_file('tasks/' + task + '.txt', step_cost=0.0)
    else:
        print('Unknown task name: ', task)
        assert(False)

    nx_graph, A, intToS, _ = getGraphFromMDP(mdp)

    experiment = {}

    global use_ASPDM
    if use_ASPDM:
        method_list = ["eigen", "fiedler", "ASPDM", "ApproxAverage"]
    else:
        method_list = ["eigen", "fiedler", "ApproxAverage"]

    for method in method_list:
        itterations_over_options_exp(experiment, method, A, num_options_list)

    # Generate Plot:

    color_dict = {  "eigen":"tab:blue",
                    "fiedler":"tab:orange",
                    "ASPDM":"tab:green",
                    "ApproxAverage":"tab:red",
                    }

    for method in experiment.keys():
        X = np.array(experiment[method]["num_ops"])
        Y = np.array(experiment[method]["itter"])
        conf = np.array(experiment[method]["conf"])
        plt.plot(X, Y, color=color_dict[method], label=type_to_name(method))
        plt.fill_between(X, Y+conf, Y-conf, color=color_dict[method], alpha=0.25)
    plt.title(task_to_title(dom,task))
    plt.xticks(np.arange(min(experiment[method]["num_ops"]), max(experiment[method]["num_ops"])+1, 1.0))
    plt.xlabel('Options')
    plt.ylabel('Average Number of Value Iterations over Goals')
    plt.legend()

    filename = f"{exp_name}_{dom}_{task}"
    plt.savefig('Plots/' + filename + '.png')
    with open('PlotData/' + filename + '.json', "w") as file:
        json.dump(experiment, file)
    plt.cla()
    plt.clf()


exp_name = "final_wconf_"
num_options_list = range(1,15)

use_ASPDM = True

# run_exp(exp_name, num_options_list, "grid", task="9x9grid_no_goal")
# run_exp(exp_name, num_options_list, "grid", task="18x18grid_no_goal")
# run_exp(exp_name, num_options_list, "grid", task="tworoom_no_goal")

# run_exp(exp_name, num_options_list, "grid", task="18x18grid")
# run_exp(exp_name, num_options_list, "grid", task="twohall")
# run_exp(exp_name, num_options_list, "grid", task="tworoom")

# run_exp(exp_name, num_options_list, "grid", task="fourroom")
# run_exp(exp_name, num_options_list, "grid", task="ParrUp")
# run_exp(exp_name, num_options_list, "taxi-S")


run_exp(exp_name, num_options_list, "hanoi")

use_ASPDM = False
num_options_list = range(1,11)

# run_exp(exp_name, num_options_list, "hanoi")
# run_exp(exp_name, num_options_list, "hanoi-L")
# run_exp(exp_name, num_options_list, "track", task="Track1")
run_exp(exp_name, num_options_list, "taxi")
run_exp(exp_name, num_options_list, "grid", task="Parr")
