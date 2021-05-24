
# Python imports.
from __future__ import print_function
import time
import argparse
import os
import math
import sys
import copy
import numpy as np
from collections import defaultdict

# Non-standard imports.
from simple_rl.planning import ValueIteration
from simple_rl.experiments import Experiment
from simple_rl.mdp import MarkovGameMDP
from simple_rl.utils import chart_utils
from simple_rl.agents import *
from simple_rl.tasks import *

from options.OptionClasses.Option import getShortestPathLengthMDP, getShortestPathLengthGraph

from tqdm import tqdm

import numpy as np
import gc
import random


def train_agents_on_mdp_online(agents, mdp, instances=10, episodes=100, steps=500, episode_sample_rate = 1, add_options_n_ep=None, num_ops_add=4):

    data_dict = {}

    for agent in agents:
        data = run_agent_on_mdp_online(agent, mdp, instances=instances, episodes=episodes, steps=steps, episode_sample_rate=episode_sample_rate, add_options_n_ep=add_options_n_ep, num_ops_add=num_ops_add)
        data_dict[agent.name] = data

    return data_dict



def run_agent_on_mdp_online(agent, mdp, instances, episodes, steps, episode_sample_rate, add_options_n_ep=None, num_ops_add=1):

    print(f"Training {agent.name}")

    instance_rewards = []
    with tqdm(total=instances * episodes) as pbar:
        for instance in range(instances):
            agent.reset()
            if hasattr(agent, 'clear_options'):
                agent.clear_options()
            episode_rewards = []

            np.random.seed(instance)
            random.seed(instance)

            mdp.reset_init_and_goal()

            for episode in range(episodes):
                if hasattr(agent, 'generate_options') and add_options_n_ep:
                    if episode != 0 and episode % add_options_n_ep == 0:
                        agent.generate_options(num_ops_add)

                mdp.reset()

                state = mdp.get_init_state()
                agent.end_of_episode()
                reward = 0
                episode_reward = 0

                agent_decisions = 0

                for step in range(1,steps+1):
                    action = agent.act(state)

                    if not hasattr(agent, 'in_option') or not agent.in_option:
                        agent_decisions += 1

                    reward, next_state = mdp.execute_agent_action(action)
                    terminal = next_state.is_terminal()
                    timeout = step == steps
                    agent.update(state, action, reward, next_state, terminal, timeout)

                    episode_reward += reward * mdp.gamma ** step

                    if terminal or timeout: #timeout happens anyway this happens anyway
                        mdp.reset_goal()
                        break
                        # mdp.reset()
                        # agent.end_of_episode()
                        # next_state = mdp.get_init_state()
                        # reward = 0 #clear reward before starting again

                    state = next_state

                if episode % episode_sample_rate == 0:
                    episode_rewards.append(episode_reward)
                    # episode_rewards.append(step)

                pbar.update(1)


            # instance_rewards.append(np.cumsum(episode_rewards))
            instance_rewards.append(episode_rewards)
            # if instance % 5 == 0:
            #     gc.collect()
    return instance_rewards
