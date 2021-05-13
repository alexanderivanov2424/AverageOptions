
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

from tqdm import tqdm

import numpy as np
import random


def train_agents_on_mdp(agents, mdp, instances=10, episodes=100, steps=500, episode_sample_rate = 1):

    data_dict = {}

    for agent in agents:
        data = run_agent_on_mdp(agent, mdp, instances=instances, episodes=episodes, steps=steps, episode_sample_rate=episode_sample_rate)
        data_dict[agent.name] = data

    return data_dict



def run_agent_on_mdp(agent, mdp, instances, episodes, steps, episode_sample_rate):

    print(f"Training {agent.name}")

    instance_rewards = []
    with tqdm(total=instances * episodes) as pbar:
        for instance in range(instances):
            agent.reset()
            episode_rewards = []

            np.random.seed(instance)
            random.seed(instance)

            try: #only valid for gridworld env
                #If random init and goal is set, a new init and goal are chosen here
                mdp.reset_init_and_goal()
            except:
                pass

            for episode in range(episodes):

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
                        break
                        # mdp.reset()
                        # agent.end_of_episode()
                        # next_state = mdp.get_init_state()
                        # reward = 0 #clear reward before starting again

                    state = next_state

                if episode % episode_sample_rate == 0:
                    # episode_rewards.append(episode_reward)
                    X = 0
                    if hasattr(agent, 'options_executed'):
                        X += agent.options_executed
                    episode_rewards.append(X)
                    # episode_rewards.append(step)

                pbar.update(1)

            # instance_rewards.append(np.cumsum(episode_rewards))
            instance_rewards.append(episode_rewards)
    return instance_rewards

def QLearn_mdp_offline( agent,
                        mdp,
                        instances=10,
                        episodes=100,
                        steps=500,
                        track_reward='sum', #sum or avg
                        reset_at_terminal=True,
                        resample_at_terminal=False,
                        cumulative_plot=True):

    value = 0
    gamma = mdp.get_gamma()

    instance_rewards = []
    for instance in range(1, instances+1):
        print("instance " + str(instance) + " of " + str(instances))
        episode_rewards = []
        # For each episode.

        try: #only valid for gridworld env
            mdp.mdp.reset_init_and_goal()
        except:
            pass

        for episode in range(1, episodes + 1):

            episode_reward = 0.0

            mdp.reset()
            state = mdp.get_init_state()
            reward = 0
            episode_start_time = time.clock()

            step = 1
            in_episode_step = 1
            while step < steps+1:
                step_start = time.clock()

                agent.actions = mdp.get_actions()
                action = agent.act(state, reward)

                # Terminal check.
                if state.is_terminal():
                    if episodes == 1 and not reset_at_terminal and experiment is not None and action != "terminate":
                        continue
                    break

                reward, next_state, length = mdp.execute_agent_action(action)

                step += length
                in_episode_step += length
                episode_reward += reward * gamma ** in_episode_step
                value += reward * gamma ** in_episode_step


                if next_state.is_terminal():
                    if reset_at_terminal:
                        # Reset the MDP.
                        mdp.reset()
                        next_state = mdp.get_init_state()
                        in_episode_step = 1

                    elif resample_at_terminal and step < steps:
                        break

                state = next_state

            if track_reward == "avg":
                episode_reward = episode_reward / step

            episode_rewards.append(episode_reward)
            action = agent.act(state, reward)

            mdp.reset()
            agent.end_of_episode()

        instance_rewards.append(episode_rewards)
    return instance_rewards
