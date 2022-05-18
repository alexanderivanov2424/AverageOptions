import pickle
import numpy as np
import gym
import d4rl 
import itertools
from PIL import Image
import xml.etree.ElementTree as ET
import os


from continuous_exp.dataCollection import loadData

import matplotlib.pyplot as plt


env = gym.make('antmaze-umaze-v2')

### custom _read_pixels_as_in_window because env.viewer._read_pixels_as_in_window() is low res
import copy
from mujoco_py.utils import rec_copy, rec_assign
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
env.viewer._read_pixels_as_in_window = _read_pixels_as_in_window
###

states_all = loadData('states_antmaze-umaze-v2')

env.render()
background = env.viewer._read_pixels_as_in_window()

blended = background.copy()


env.reset()


for i, state in enumerate(states_all):

    env.set_state(state[0], state[1])

    if i % 5 == 0 and i > 0:
        image = env.viewer._read_pixels_as_in_window()
        blended[background!=image] = image[background!=image]
    
    plt.imshow(blended)
    plt.draw()
    plt.pause(.01)
    plt.cla()

Image.fromarray(blended).save('test_medium.pdf')

