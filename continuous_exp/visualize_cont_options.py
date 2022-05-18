from re import X
from continuous_exp.dataCollection import loadData, buildAdjacencyFromStates, visualizeStates

import gym
import d4rl 
import numpy as np

from continuous_exp.search import *
from continuous_exp.ballOptions import *
from continuous_exp.utils import *

from continuous_exp.dataCollection import sampleStates, sampleStatesFromRoots, saveData, loadData

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
from PIL import Image

import copy
import mujoco_py
from mujoco_py.utils import rec_copy, rec_assign

matplotlib.style.use('default')
matplotlib.use('TkAgg')



def viewer_setup(env):
    cam = env.viewer.sim._render_context_window.cam
    #cam.trackbodyid = -1         # id of the body to track ()
    cam.distance =  20
    cam.lookat[0] = 4         # x,y,z offset from the object (works if trackbodyid=-1)
    cam.lookat[1] = 4
    cam.lookat[2] = 1
    cam.elevation = -90           # camera rotation around the axis in the plane going through the frame origin (if 0 you just see a line)
    cam.azimuth = 90              # camera rotation around the camera's vertical axis


"""
method = 'eigen' 'fiedler' 'ASPDM' 'ApproxAverage' 'hitting' 'none'

"""
def buildOptions(states, A, num_ops, method, R = 1):

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

def plot_option(save_name, env, option, states):
    
    def _read_pixels_as_in_window(resolution = (2200,2000)):
        if env.viewer.sim._render_context_offscreen is None:
            env.viewer.sim.render(*resolution)
        offscreen_ctx = env.viewer.sim._render_context_offscreen
        window_ctx = env.viewer.sim._render_context_window
        # Save markers and overlay from offscreen.
        saved = [copy.deepcopy(offscreen_ctx._markers),
                 copy.deepcopy(offscreen_ctx._overlay),
                 rec_copy(offscreen_ctx.cam)]
        # Copy markers and overlay from window.
        offscreen_ctx._markers[:] = window_ctx._markers[:]
        offscreen_ctx._overlay.clear()
        offscreen_ctx._overlay.update(window_ctx._overlay)
        rec_assign(offscreen_ctx.cam, rec_copy(window_ctx.cam))

        img = env.viewer.sim.render(*resolution)
        # Restore markers and overlay to offscreen.
        offscreen_ctx._markers[:] = saved[0][:]
        offscreen_ctx._overlay.clear()
        offscreen_ctx._overlay.update(saved[1])
        rec_assign(offscreen_ctx.cam, saved[2])
        return np.flip(img,0) 

    env.render()
    viewer_setup(env)
 
    env.viewer._read_pixels_as_in_window = _read_pixels_as_in_window

    env.set_xy((12,12))
    env.render()
    background = env.viewer._read_pixels_as_in_window()

    blended = background.copy()

    env.reset()

    for i, state in enumerate(states):
        setEnvData(env, state)
        print(i/len(states), end='\r')
        if i % 100 == 0 and i > 0:
            if option.canRun(state):
                image = env.viewer._read_pixels_as_in_window()
                blended[background!=image] = image[background!=image]

                # plt.imshow(blended)
                # plt.draw()
                # plt.pause(.01)
                # plt.cla()
                # plt.clf()

                option.run(env)
                image = env.viewer._read_pixels_as_in_window()
                blended[background!=image] = image[background!=image]

    Image.fromarray(blended).save(f"./continuous_exp/plots/optionPlots/{save_name}.png")

def visualize_options(env_name, states_disc, states_all, method, num_options, op_i):
    np.random.seed(0)
    random.seed(0)

    

    A = buildAdjacencyFromStates(states_disc, .5, 10, use_2D=True)

    options = buildOptions(states_disc, A, num_options, method)

    env = gym.make(env_name)
    
    op = options[op_i]        
    env.reset()
    plot_option(f"option_visual_dense_{method}_{op_i}", env, op, states_all)
        

import sys
method = sys.argv[1]
num_ops = int(sys.argv[2])
op_i = int(sys.argv[3])
r = int(sys.argv[4])


ENV_NAME = 'antmaze-umaze-v2'

states_disc = loadData(f'disc_r{r}_states_antmaze-umaze-v2')
states_all = loadData('states_antmaze-umaze-v2')
visualize_options(ENV_NAME, states_disc, states_all, method, num_ops, op_i)