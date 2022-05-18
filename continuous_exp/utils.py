import numpy as np
import os
import pickle
import json

import matplotlib.pyplot as plt

from continuous_exp.continuousExperiment import Experiment

# Env Human Readible Util
#===========================================================================================================================

def getEnvPos(env):
    env_name = getEnvName(env)
    if env_name == "antmaze-umaze-v2":
        return getEnvVecIdx(env, [0,1])
    elif env_name == "FetchReach-v1":
        return np.copy(env.unwrapped.sim.data.get_site_xpos("robot0:grip"))

def vecToPos(env, vec, use_pos=False, idx=None):
    if (not use_pos) and (idx is None):
        return vec
    elif use_pos and (not idx is None):
        assert(False) #"Cannot specify both idx and use_pos arguments")
    elif use_pos:
        setEnvVec(env, vec)
        return getEnvPos(env)
    elif not idx is None:
        return vec[idx]


# Env State and Vector Utils
#===========================================================================================================================

def getEnvState(env):
    return env.unwrapped.sim.get_state()

def setEnvState(env, state):
    env.unwrapped.sim.set_state(state)
    env.unwrapped.sim.forward()

def stateToVec(state):
    return np.concatenate((state.qpos, state.qvel))

def getVecIdx(vec, idx):
    return vec[idx]

def getEnvVec(env):
    return stateToVec(getEnvState(env))

def setEnvVec(env, vec):
    env.unwrapped.sim.set_state_from_flattened(np.concatenate(([0], vec)))
    env.unwrapped.sim.forward()

def getEnvVecIdx(env, idx):
    v = getEnvVec(env)
    if idx is None:
        return v
    return v[idx]

def setEnvVecIdx(env, values, idx):
    if idx is None:
        setEnvVec(env, values)
        return
    
    v = getEnvVec(env)
    for i,val in zip(idx,values):
        v[i] = val
    setEnvVec(env, v)


# Saving and Loading Utils
#===========================================================================================================================

def getEnvName(env):
    return env.unwrapped.spec.id

DIRECTORY = "./continuous_exp/data_and_plots"

DATA_SAVE = "data"
EXPERIMENT_SAVE = "experiments"
PLOT_SAVE = "plots"
TRAJECTORY_SAVE = "trajectories"





def saveData(env_name, file_name, data):
    path = os.path.join(DIRECTORY, env_name, DATA_SAVE, file_name) + ".pkl"
    with open(path, 'wb') as file:
        pickle.dump(data, file) 

def loadData(env_name, file_name):
    path = os.path.join(DIRECTORY, env_name, DATA_SAVE, file_name) + ".pkl"
    with open(path, 'rb') as file:
        data = pickle.load(file)
    return data

def saveExperiment(env_name, experiment):
    path = os.path.join(DIRECTORY, env_name, EXPERIMENT_SAVE, experiment.exp_name) + ".pkl"
    with open(path, "w") as file:
        json.dump(experiment.exp, file)

def loadExperiment(env_name, exp_name):
    experiment = Experiment(exp_name)
    path = os.path.join(DIRECTORY, env_name, EXPERIMENT_SAVE, exp_name) + ".pkl"
    with open(path, "r") as file:
        experiment.exp = json.load(file)
    experiment.exp_name = experiment.exp['exp_name']
    return experiment


def saveFig(env_name, file_name):
    path = os.path.join(DIRECTORY, env_name, PLOT_SAVE, file_name) + ".png"
    plt.savefig(path)


def saveTrajectory(env_name, file_name, trajectory):
    path = os.path.join(DIRECTORY, env_name, TRAJECTORY_SAVE, file_name) + ".pkl"
    with open(path, 'wb') as file:
        pickle.dump(trajectory, file)

def loadTrajectory(env_name, file_name):
    path = os.path.join(DIRECTORY, env_name, TRAJECTORY_SAVE, file_name) + ".pkl"
    with open(path, 'rb') as file:
        trajectory = pickle.load(file)
    return trajectory
    

# Old
#===========================================================================================================================

# def getEnvData(env):
#     env_name = env.unwrapped.spec.id
#     if env_name == 'antmaze-umaze-v2':
#         return np.concatenate((np.copy(env.physics.data.qpos), np.copy(env.physics.data.qvel)))
#     return None


# def setEnvData(env, data):
#     env_name = env.unwrapped.spec.id
#     if env_name == 'antmaze-umaze-v2':
#         env.set_state(data[0:15], data[15:])
#     return None


# def getPos(env):
#     env_name = env.unwrapped.spec.id
#     if env_name == 'antmaze-umaze-v2':
#         return np.array([env.get_body_com("torso")[0], env.get_body_com("torso")[1]])
#     return None

# def setPos(env, pos):
#     env_name = env.unwrapped.spec.id
#     if env_name == 'antmaze-umaze-v2':
#         env.set_xy(pos)


def inRange(v1, v2, r):
    return np.sqrt(np.sum(np.square(v1 - v2))) < r