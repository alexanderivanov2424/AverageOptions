from fileinput import filename
import heapq
from collections import defaultdict
import matplotlib
from matplotlib.pyplot import getp
from matplotlib.style import available
import numpy as np

from continuous_exp.dataCollection import saveData, loadData

from continuous_exp.utils import getEnvPos, getEnvName, getEnvState, setEnvState, getEnvVec, setEnvVec, getEnvVecIdx, setEnvVecIdx, inRange
from continuous_exp.utils import saveTrajectory, loadTrajectory

def reward(pos, g_pos):
    return - np.sqrt(np.sum(np.square(g_pos - pos)))

def discounted_reward(R, gamma = .95):
    return np.sum(np.power(gamma, np.arange(len(R))) * np.array(R))

def sample_trajectory(env, g_pos, traj_length, samples):
    start_state = getEnvState(env)

    best_trajectory = []
    best_trajectory_dist = np.sum(np.square(getEnvPos(env) - g_pos))

    for sample in range(samples):
        actions = []
        setEnvState(env, start_state)

        for i in range(traj_length):
            a = env.action_space.sample()
            if sample == 0:
                a *= 0
            actions.append(a)
            env.step(a)
            #env.render(mode = "human")
        
        d = np.sum(np.square(getEnvPos(env) - g_pos))
        if d < best_trajectory_dist:
            best_trajectory = actions
            best_trajectory_dist = d


    return best_trajectory


"""
g_pos - goal position. Must correspond to values specified b idx. If idx is None then this is the full state vector.
"""
def sample_trajectory_options(env, g_pos, range_to_goal, options, traj_length, samples, use_pos=False, idx=None, option_prob=.5):
    start_state = getEnvState(env)

    def getAvailOptions(env):
        return [op for op in options if op.canRun(env, getEnvVec(env), use_pos=use_pos, idx=idx)]

    best_trajectory = []
    best_trajectory_return = - np.inf

    for sample in range(samples):
        actions = []
        rewards = []
        setEnvState(env, start_state)

        for i in range(traj_length):
            available_options = getAvailOptions(env)
            if len(available_options) > 0 and np.random.rand() < option_prob:
                op = np.random.choice(available_options)
                actions.append(op)
                op.run(env)
            else:
                a = env.action_space.sample()
                if sample == 0:
                    a *= 0
                actions.append(a)
                env.step(a)
            
            pos = getEnvPos(env)
            if inRange(pos, g_pos, range_to_goal):
                return actions, traj_length * sample + i

            rewards.append(reward(pos, g_pos))
        
        traj_return = discounted_reward(rewards)
        if traj_return > best_trajectory_return:
            best_trajectory = actions
            best_trajectory_return = traj_return

    return best_trajectory, samples * traj_length


def plan_to_goal(env, g_pos, range_to_goal, options, use_pos=False, idx=None, max_iter=10000):
    start_state = getEnvState(env)
    planning_steps = 0
    options_used = 0

    traj_length = 5
    samples = 400

    pos = getEnvPos(env)

    trajectory = []

    while planning_steps < max_iter and not inRange(pos, g_pos, range_to_goal):
        start_state = getEnvState(env)

        best_trajectory, steps = sample_trajectory_options(env, g_pos, range_to_goal, options, traj_length, samples, use_pos=use_pos, idx=idx)
        trajectory.extend(best_trajectory)
        planning_steps += steps

        setEnvState(env, start_state)

        for a in best_trajectory:
            try:
                env.step(a)
            except:
                options_used += 1
                a.run(env) 
            #env.render(mode = "human")
        
        pos = getEnvPos(env)

    return planning_steps, options_used, trajectory

def plan_from_pos_to_pos(env, s_pos, g_pos, range_to_goal, options, use_pos=False, idx=None, max_iter=10000, exp_name="trajectory"):
    setEnvVecIdx(env, s_pos, idx)
    if use_pos:
        idx = None
    planning_steps, options_used, trajectory = plan_to_goal(env, g_pos, range_to_goal, options, use_pos=use_pos, idx=idx, max_iter=max_iter)

    env_name = getEnvName(env)
    file_name = f"{exp_name}_{s_pos}_to_{g_pos}_op_{len(options)}"
    saveTrajectory(env_name, file_name, trajectory)

    return planning_steps, options_used

def visualize_trajectory(env, s_pos, g_pos, num_options, exp_name="trajectory"):
    env_name = getEnvName(env)
    file_name = f"{exp_name}_{s_pos}_to_{g_pos}_op_{num_options}"
    trajectory = loadTrajectory(env_name, file_name)
    for a in trajectory:
        try:
            env.step(a)
        except:
            a.run(env) 
        #print(getEnvVecIdx(env, [0,1]))
        env.render(mode = "human")

# class State:
#     def __init__(self, data, g):
#         self.data = data
#         self.g = g
#         self.added_score = 0

#     def __lt__(self, other):
#         p1 = np.array(self.data[0][:2])
#         p2 = np.array(other.data[0][:2])
#         return np.sum(np.square(p1 - self.g)) + self.added_score < np.sum(np.square(p2 - self.g)) + other.added_score

#     def __eq__(self, other):
#         return self.data == other.data


# def setEnvData(env_state, state):
#     env_state.set_state(np.array(state.data[0]), np.array(state.data[1]))

# def getStateData(state):
#     return tuple(state.physics.data.qpos), tuple(state.physics.data.qvel)

# def reconstructPath(cameFrom, actionsTaken, current):
#     actions = []
#     while current in cameFrom.keys():
#         actions.insert(0, actionsTaken[current])
#         current = cameFrom[current]
#     return actions

# def astar(env, g_pos, delta):

#     def getState(env_state):
#         return State(getStateData(env_state), g_pos)

#     def h(state):
#         p = np.array(state.data[0][:2])
#         return np.sum(np.square(p - g_pos))

#     start = getState(env)

#     openSet = [start]
#     heapq.heapify(openSet)

#     cameFrom = defaultdict(lambda : np.Inf) # data of state before given state data in shortest path
#     actionsTaken = dict() #action take to get to the given state data in shortest path

#     gScore = defaultdict(lambda : np.Inf) # cost of shortest path from start state to given state data
#     gScore[start.data] = 0

#     while len(openSet) > 0:

#         current = heapq.heappop(openSet)
#         if h(current) <= delta:
#             print(current.data[0][:2])
#             setEnvData(env, start)
#             return reconstructPath(cameFrom, actionsTaken, current.data)


#         for i in range(2):
#             setEnvData(env, current)
#             a = env.action_space.sample()
#             env.step(a)
#             neighbor = getState(env)
#             tentative_score = gScore[current.data] + 1
#             if tentative_score < gScore[neighbor.data]:
#                 cameFrom[neighbor.data] = current.data
#                 actionsTaken[neighbor.data] = a
#                 gScore[neighbor.data] = tentative_score

#                 neighbor.added_score = tentative_score
#                 if not neighbor in openSet and len(openSet) < 100:
#                     heapq.heappush(openSet, neighbor)

#     setEnvData(env, start)
#     assert(False, "A* Failed")



# def setStateData(s,data):
#     s.set_state(*data)

# def h(pos, goal):
#     return np.sum(np.square(pos - goal))




# def sampleAction(s, g):
#     start_data = getStateData(s)

#     nearest_action = None 
#     nearest_position = getPos(s)
    
#     for _ in range(50):
#         setStateData(s, start_data)
#         a = s.action_space.sample()
#         s.step(a)
#         new_pos = getPos(s)
#         if nearest_action is None or h(new_pos, g) < h(nearest_position, g):
#             nearest_action = a 
#             nearest_position = new_pos

#     setStateData(s, start_data)
#     return nearest_action

    
    