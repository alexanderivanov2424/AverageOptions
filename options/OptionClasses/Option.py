import networkx as nx
from collections import deque
import matplotlib.pyplot as plt
import numpy as np
import copy

class Option:
    def __init__(self, start, end, policyDict, point_option=True):
            self.start = start
            self.end = end
            self.policy = policyDict
            self.point_option = point_option

    def is_termination_state(self, state):
        return state == self.end

    def can_run(self, state):
        if self.point_option:
            return state == self.start
        return state in self.policy.keys()

    def act(self, state):
        return self.policy[state]


def constructPointOptionObject(option_i_pair, graph, intToS):
    start = intToS[option_i_pair[0]]
    end = intToS[option_i_pair[1]]
    path = nx.shortest_path(graph, source=start, target=end)
    policyDict = {}
    for i in range(len(path)-1):
        policyDict[path[i]] = graph[path[i]][path[i+1]][0]['action']
    return Option(start, end, policyDict, point_option=True)

def constructOptionObject(option_i_pair, graph, intToS):
    start = intToS[option_i_pair[0]]
    end = intToS[option_i_pair[1]]
    path = nx.shortest_path(graph, target=end)
    policyDict = {}
    for start in path.keys():
        if start == end:
            continue
        policyDict[start] = graph[path[start][0]][path[start][1]][0]['action']
    return Option(start, end, policyDict, point_option=False)

def getShortestPathLengthMDP(mdp):
    graph, _, _, terminal = getGraphFromMDP(mdp)
    end = None
    length = None
    for end in terminal:
        l = len(nx.shortest_path(graph, source=mdp.init_state, target=end))
        if length == None or l < length:
            length = l
    return length

def getShortestPathLengthGraph(mdp, graph):
    terminal = []
    for s in graph.nodes:
        for a in mdp.get_actions():
            s_ = mdp.transition_func(s,a)
            r =  mdp.reward_func(s,a,s_)
            if r > 0:
                terminal.append(s)
                break
        if len(terminal) > 0:
            break
    end = None
    length = None
    for end in terminal:
        l = len(nx.shortest_path(graph, source=mdp.init_state, target=end))
        if length == None or l < length:
            length = l
    return length

def getShortestPathLengthGraphMany(mdp, graph):
    end = None
    for s in graph.nodes:
        for a in mdp.get_actions():
            s_ = mdp.transition_func(s,a)
            r =  mdp.reward_func(s,a,s_)
            if r > 0:
                end = s
                break
        if end:
            break

    paths = nx.shortest_path(graph, target=end)
    lengths = {}
    for key in paths.keys():
        lengths[key] = len(paths[key])
    return lengths

def getGraphFromMDP(mdp):
    G = nx.MultiDiGraph()
    terminal = []
    queue = deque()

    mdp.reset()
    state = copy.deepcopy(mdp.cur_state)
    G.add_node(state)

    ACTIONS = mdp.get_actions()
    for a in ACTIONS:
        queue.append([a])

    while len(queue) > 0:
        mdp.reset()
        state = copy.deepcopy(mdp.cur_state)
        prev_state = state
        action_list = queue.popleft()

        for a in action_list:
            prev_state = state
            reward, state = mdp.execute_agent_action(a)
            state = copy.deepcopy(state)
            if reward != 0:
                terminal.append(state)

        if state in G.nodes:
            G.add_edge(prev_state,state, action = action_list[-1])
            # G.add_edge(prev_state,state)
            # G[prev_state][state]['action'] = action_list[-1]
            continue

        G.add_edge(prev_state,state, action = action_list[-1])
        # G[prev_state][state]['action'] = action_list[-1]


        for a in ACTIONS:
            queue.append(list(action_list) + [a])

    # nx.draw_spectral(G)
    # plt.show()

    intToS = {}
    for i, n in enumerate(G.nodes):
        intToS[i] = n
    return G, nx.to_numpy_matrix(G), intToS, terminal

def getGraphFromExp(experiences, nx_graph):
    states = set()

    for s, a, s_ in experiences:
        states.add(s)
        states.add(s_)

    G = nx_graph.subgraph(states)

    intToS = {}
    for i, n in enumerate(G.nodes):
        intToS[i] = n
    return G, nx.to_numpy_matrix(G), intToS

def getDiGraphFromExp(experiences):
    G = nx.DiGraph()
    terminal = []

    for s, a, s_ in experiences:
        G.add_edge(s,s_)
        G[s][s_]['action'] = a

    intToS = {}
    for i, n in enumerate(G.nodes):
        intToS[i] = n
    return G, nx.to_numpy_matrix(G), intToS
