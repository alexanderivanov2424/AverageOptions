import json
import numpy as np
import scipy.stats

import matplotlib
import matplotlib.path as mpath
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

def plot_exp(prefix, exp_name, n_instances, n_episodes, n_options):
    filename = f'{prefix}_{exp_name}_inst{n_instances}_ep{n_episodes}_op{n_options}'

    with open('PlotData/' + filename + '.json', "r") as file:
        experiment = json.load(file)


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
        data = np.apply_along_axis(lambda x: np.convolve(x, np.ones(10)/10, "valid"), 1, data)
        Y = np.mean(data, axis=0)
        se = scipy.stats.sem(data, axis=0)
        conf = se * scipy.stats.t.ppf((1 + .8) / 2., len(Y)-1)
        plt.fill_between(range(len(Y)), Y + conf, Y - conf, color=color_dict[agent_name], alpha=0.25)
        plt.plot(range(len(Y)), Y, color=color_dict[agent_name], label=agent_name)

    # plt.title(exp_name)
    plt.xlabel('episode')
    plt.ylabel('fraction of maximum return')
    plt.legend()
    plt.show(block=True)


# plot_exp('offline_TEST', '9x9grid', 100, 100, 8)
# plot_exp('offline_TEST', 'fourroom', 100, 100, 8)
# plot_exp('offline_TEST', 'hanoi', 100, 100, 8)
# plot_exp('offline_TEST', 'track', 100, 100, 8)
# plot_exp('offline_TEST', 'taxi', 100, 100, 8)
# plot_exp('offline_TEST', 'Parr', 100, 200, 16)

plot_exp('online_TEST', '9x9grid', 100, 300, 2)
plot_exp('online_TEST', 'fourroom', 100, 300, 2)
plot_exp('online_TEST', 'hanoi', 100, 300, 2)
plot_exp('online_TEST', 'track', 100, 300, 2)
# plot_exp('online_TEST', 'taxi', 100, 300, 2)
# plot_exp('online_TEST', 'Parr', 100, 300, 2)

exp_name = 'fourroom'
n_instances = 100
n_episodes = 100
n_options = 8
