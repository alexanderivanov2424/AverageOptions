from re import X
from continuous_exp.dataCollection import loadData, buildAdjacencyFromStates, visualizeStates

import gym
import d4rl 
import numpy as np

from continuous_exp.search import *
from continuous_exp.ballOptions import *
from continuous_exp.utils import getEnvPos
from continuous_exp.utils import saveExperiment, loadExperiment, saveFig

# from continuous_exp.dataCollection import 
from continuous_exp.continuousExperiment import Experiment

from options.FiedlerOptions import FiedlerOptions
from options.EigenOptions import Eigenoptions
from options.AverageOptions import AverageShortestOptions
from options.ApproxAverageOptions import ApproxAverageOptions
from options.HittingOptions import HittingTimeOptions

from options.OptionGeneration import GetOptions

import os
import networkx as nx
import numpy as np
import random
import json
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm

matplotlib.style.use('default')
matplotlib.use('TkAgg')


color_dict = {  "eigen":"tab:blue",
                "fiedler":"tab:orange",
                "ASPDM":"tab:green",
                "ApproxAverage":"tab:red",
                }

def type_to_name(type):
    if type == "eigen":
        return "Eigen Options"
    if type == "fiedler":
        return "Covering Options"
    if type == "ASPDM":
        return "Average Options"
    if type == "ApproxAverage":
        return "Fast Average Options"

"""
method = 'eigen' 'fiedler' 'ASPDM' 'ApproxAverage' 'hitting' 'none'

"""
def buildOptions(states, A, num_ops, method, R):

    if method == 'none':
        return []

    B, option_i_pairs, _ = GetOptions(A, num_ops, method)

    options = []
    for op_i in option_i_pairs:
        rootData = states[op_i[0]]
        endData = states[op_i[1]]
        op = Option(rootData, endData, R)
        options.append(op)

    return options


def measure_mean_planning_steps(experiment, env, states, A, method, num_options, sources, targets, option_r, range_to_goal, use_pos=False, idx=None):
    env.reset()

    options = buildOptions(states, A, num_options, method, option_r)

    experiment.start_method(method)

    Distance = []
    Planning_Steps = []
    Options_Used = []

    max_iter = 10**4
    
    c = 0
    c_total = len(sources) * len(targets)

    for i, s_pos in enumerate(sources):
        for j, g_pos in enumerate(targets):
            # if i == j:
            #     continue
            c += 1
            print("Start: ", s_pos, "Goal: ", g_pos, f"{c} / {c_total}")#, end='\r')
            steps, options_used = plan_from_pos_to_pos(env, s_pos, g_pos, range_to_goal, options, use_pos=use_pos, idx=idx, max_iter=max_iter, exp_name=experiment.exp_name)

            print(steps, options_used)
            if len(s_pos) == len(g_pos):
                d = np.sqrt(np.sum(np.square(np.array(s_pos) - np.array(g_pos))))
            else:
                d = 0
            Distance.append(d)
            Planning_Steps.append(steps)
            Options_Used.append(options_used)

            experiment.log_planning_run(method, num_options, s_pos, g_pos, d, steps, options_used)

    print(f'Finished Run {method} {num_options}', ' '*50)
    experiment.log_average_planning_steps(method, num_options, np.mean(Planning_Steps))


def run_experiment(exp_name, env_name, data_path, method_list, num_options_list, sources, targets, eps, knn, option_r, range_to_goal, use_pos_adj=False, idx_adj=None, use_pos=False, idx=None, plot=True, save_exp=True):
    np.random.seed(0)
    random.seed(0)

    env = gym.make(env_name)

    states = loadData(env_name, data_path)
    A = buildAdjacencyFromStates(env, states, eps, knn, use_pos=use_pos_adj, idx=None if use_pos_adj else idx_adj)

    # visualizeStates(states, A)
    

    experiment = Experiment(exp_name)
    experiment.set_env_name(env_name)

    for method in method_list:
        for num_options in num_options_list:
            print(f"[+] Running {method} with {num_options} options")
            measure_mean_planning_steps(experiment, env, states, A, method, num_options, sources, targets, option_r, range_to_goal, use_pos=use_pos, idx=idx)
    
    if save_exp:
        saveExperiment(env_name, experiment)

    if not plot:
        return experiment

    for method in method_list:
        X = []
        Y = []
        for num_options in num_options_list:
            X.append(num_options)
            Y.append(experiment.get_average_planning_steps(method, num_options))

        plt.plot(X,Y, color=color_dict[method], label=type_to_name(method))

    plt.title("Average Planning Steps versus Number of Options")
    plt.xticks(np.arange(min(num_options_list), max(num_options_list)+1, 1.0))
    plt.xlabel("Number of Options")
    plt.ylabel("Planning Steps (# of virtual steps)")
    plt.legend()
    saveFig(env_name, exp_name)
    #plt.show()
    plt.cla()
    plt.clf()
    return experiment

def plotLine(env_name, exp_name, method, color, label, line_style):
    exp = loadExperiment(env_name, exp_name)
    X, Y = exp.get_options_and_average_planning_steps(method)
    X = [float(x) for x in X]
    Y = [float(y) for y in Y]
    plt.plot(X,Y, color=color, label=label, ls = line_style)

def all_disc_ApproxAverage(env_name):
    color_map = cm.Reds(np.linspace(.3,.9,8))
    line_style = ['-', '--', '-.', ':'] * 3

    for i,r in enumerate([6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10]):
        EXP_NAME = f'disc_r{r}_full'
        plotLine(env_name, EXP_NAME, 'ApproxAverage', color_map[i], f'Fast Average Options - {r}', line_style[i])

    plt.title("Average Planning Steps versus Number of Options")
    plt.xticks(np.arange(2, 16+1, 1.0))
    plt.xlabel("Number of Options")
    plt.ylabel("Planning Steps (# of virtual steps)")
    plt.legend()
    saveFig(env_name, "ApproxAverage_all_disc_full")
    plt.show()

def all_disc_ASPDM(env_name):
    color_map = cm.Greens(np.linspace(.5,1,5))
    line_style = ['-', '--', '-.', ':'] * 2

    for i,r in enumerate([8, 8.5, 9, 9.5, 10]):
        EXP_NAME = f'disc_r{r}_full'
        plotLine(env_name, EXP_NAME, 'ASPDM', color_map[i], f'Average Options - {r}', line_style[i])

    plt.title("Average Planning Steps versus Number of Options")
    plt.xticks(np.arange(2, 16+1, 1.0))
    plt.xlabel("Number of Options")
    plt.ylabel("Planning Steps (# of virtual steps)")
    plt.legend()
    saveFig(env_name, "ASPDM_all_disc_full")
    plt.show()

def plot_exp(env_name, exp_name):
    exp = loadExperiment(env_name, exp_name)

    n_methods = len(list(exp.exp['methods'].keys()))
    n_ops = 3
    n_runs = 156

    Data = np.zeros((n_methods, n_ops, n_runs))

    for i, method in enumerate(list(exp.exp['methods'].keys())):
        for j,num_options in enumerate(list(exp.exp['methods'][method]['avg_planning_steps'].keys())):
            runs = [r for r in exp.exp['methods'][method]['runs'] if float(r['num_options']) == float(num_options)]
            for k, run in enumerate(runs):
                Data[i, j, k] = float(run['planning_steps'])

    conf = np.std(Data, axis=2)/(n_runs**.5)
    
    for i, method in enumerate(list(exp.exp['methods'].keys())):
        X, Y = exp.get_options_and_average_planning_steps(method)
        X = [float(x) for x in X]
        Y = [float(y) for y in Y]
        plt.plot(X,Y, color=color_dict[method], label=type_to_name(method))
        plt.fill_between(X, Y + conf[i], Y - conf[i], color=color_dict[method], alpha=0.25)

    num_options_list = list(exp.exp['methods'][method]['avg_planning_steps'].keys())
    num_options_list = [int(x) for x in num_options_list]
    plt.title(f"Average Planning Steps versus Number of Options\n{env_name}")
    plt.xticks(np.arange(min(num_options_list), max(num_options_list)+1, 1.0))
    plt.xlabel("Number of Options")
    plt.ylabel("Planning Steps (# of virtual steps)")
    plt.legend()
    saveFig(env_name, exp_name)
    #plt.show()
    plt.cla()
    plt.clf()


if __name__ == "__main__":
    ENV_NAME = 'antmaze-umaze-v2'
    use_pos=True
    IDX = [0,1]
    disc_factors = [8]#[6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10]
    #targets = [(0,0), (4,0), (8,0), (8,4), (8,8), (4,8), (0,8)]
    targets = [(0,0), (2,0), (4,0), (6,0), (8,0), (8,2), (8,4), (8,6), (8,8), (6,8), (4,8), (2,8), (0,8)]
    # targets = [(0,0), (8,0), (8,8), (0,8)]
    sources = targets
    eps = .5
    knn = 10
    option_r = 1
    range_to_goal = 1
    use_pos_adj = use_pos
    idx_adj = IDX


    # ENV_NAME = "FetchReach-v1"
    # use_pos=True
    # IDX = [6,7]
    # disc_factors = [.1, .2, .3]
    # x = .5
    # sources = [(-x, -x), (-x, 0), (-x, x), (0, -x), (0, 0), (0, x), (x, -x), (x, 0), (x, x)]
    # targets = [[1, .5, .25], [1, 1, .25], [1, 1.25, .25], 
    #     [1, .5, .5], [1, 1, .5], [1, 1.25, .5], 
    #     [1, .5, .75], [1, 1, .75], [1, 1.25, .75]]#,
    #     #[.5, .25, .5], [.5, .75, .5], [.5, 1.25, .5]]
    # eps = [r + .05 for r in disc_factors]
    # knn = 2
    # option_r = .1
    # range_to_goal = .2
    # use_pos_adj = True
    # idx_adj = None

    # for r in disc_factors:
    #     plot_exp(ENV_NAME, f'disc_r{r}_full')

    # exit()

    # all_disc_ApproxAverage(ENV_NAME)
    # all_disc_ASPDM(ENV_NAME)

    for i, r in enumerate(disc_factors):
        EXP_NAME = f'disc_r{r}_full'#TEST_large'


        data_path = f'disc_r{r}_states'#_pos'

        method_list = ['eigen', 'fiedler', 'ApproxAverage']
        if r >= 8:
            method_list.append('ASPDM')
        num_options_list = [4, 8, 16]

        e = eps
        if type(eps) == list:
            e = eps[i]

        run_experiment(EXP_NAME, ENV_NAME, data_path, method_list, num_options_list, sources, targets, e, knn, option_r, range_to_goal, use_pos_adj=use_pos_adj, idx_adj=idx_adj, use_pos=use_pos, idx=IDX, plot=True, save_exp=True)

    