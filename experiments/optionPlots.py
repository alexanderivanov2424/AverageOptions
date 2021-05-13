from simple_rl.tasks import GridWorldMDP, TaxiOOMDP, HanoiMDP#, GymMDP
from simple_rl.tasks.grid_world.GridWorldMDPClass import make_grid_world_from_file

# options
from options.FiedlerOptions import FiedlerOptions
from options.EigenOptions import Eigenoptions
from options.AverageOptions import AverageShortestOptions
from options.ApproxAverageOptions import ApproxAverageOptions
from options.MinimumHittingTimeOptions import MinimumHittingOptions

from options.graph.mdp import GetAdjacencyMatrix

import matplotlib
import matplotlib.path as mpath
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

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
    elif method == 'MinHitting':
        B, options = MinimumHittingOptions(A, [len(A)-1], k)
        vectors = None

    return B, options, intToS, vectors


"""
grid mdp only
"""
def plotOptions(mdp, intToS, options, type="eigen"):
    fig, ax = plt.subplots()

    size_x = 0
    size_y = 0
    states = set()
    for k in intToS.keys():
        s = intToS[k]
        size_x = max(s.x,size_x)
        size_y = max(s.y,size_y)
        states.add((s.x,s.y))

    for i in range(size_x):
        for j in range(size_y):
            rect = mpatches.Rectangle((i,j),1,1)
            rect.set_fill(not (i+1,j+1) in states)
            rect.set_facecolor((.5,.5,.5))
            rect.set_edgecolor((0,0,0))
            ax.add_patch(rect)

    for option in options:
        start = intToS[option[0]]
        end = intToS[option[1]]
        start_x = start.x - .5
        start_y = start.y - .5
        end_x = end.x - .5
        end_y = end.y - .5
        ax.plot(start_x,start_y, "ro")
        ax.plot(end_x,end_y, "ro")

        ax.annotate(text='', xy=(end_x,end_y), xytext=(start_x,start_y), arrowprops=dict(arrowstyle='<->'))
    ax.set_title(type + ' options')

    plt.show(block=True)

def plotOptionGraphs(matrix, type="eigen"):
    fig, ax = plt.subplots()
    nx.draw_spectral(nx.to_networkx_graph(matrix),ax=ax, node_size=50, node_color=(1,0,0))

    ax.set_title(type + 'Graph with ' + type + ' options')

    plt.show(block=True)



if __name__ == '__main__':
    n_options = 3
    dom = "grid"
    task = "5x5grid"

    if dom == 'grid':
        mdp = make_grid_world_from_file('tasks/' + task + '.txt')
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

    # fiedlerMatrix, foptions, _, _ = GetOption(mdp, n_options, matrix=origMatrix, intToS=intToS, method='fiedler')
    # eigenMatrix, eoptions, _, _ = GetOption(mdp, n_options, matrix=origMatrix, intToS=intToS, method='eigen')
    ASPDMMatrix, ASPDMoptions, _, _ = GetOption(mdp, n_options, matrix=origMatrix, intToS=intToS, method='ASPDM')
    # ApproxAverageMatrix, ApproxAverageoptions, _, _ = GetOption(mdp, n_options, matrix=origMatrix, intToS=intToS, method='ApproxAverage')
    # MinHittingMatrix, MinHittingoptions, _, _ = GetOption(mdp, n_options, matrix=origMatrix, intToS=intToS, method='MinHitting')
    # _, boptions, _, _ = GetOption(mdp, n_options, matrix=origMatrix, intToS=intToS, method='bet')


    # print(foptions)
    # print(eoptions)
    print(ASPDMoptions)
    # print(ApproxAverageoptions)
    # print(boptions)
    # print(MinHittingoptions)

    # plotOptions(mdp, intToS, foptions, "Covering")
    # plotOptions(mdp, intToS, eoptions, "Eigen")
    plotOptions(mdp, intToS, ASPDMoptions, "Average")
    # plotOptions(mdp, intToS, ApproxAverageoptions, "ApproxAverage")
    # plotOptions(mdp, intToS, MinHittingoptions, "MinHitting")

    # plotOptionGraphs(fiedlerMatrix, "Covering")
    # plotOptionGraphs(eigenMatrix, "Eigen")
    # plotOptionGraphs(ASPDMMatrix, "Average")
    # plotOptionGraphs(ApproxAverageMatrix, "ApproxAverage")
