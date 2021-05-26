''' QLearningAgentClass.py: Class for a basic QLearningAgent '''

# Python imports.
import random
import numpy
import time
from collections import defaultdict

from options.OptionClasses import Option

# Other imports.
from simple_rl.agents.AgentClass import Agent

class OptionAgent(Agent):
    ''' Implementation for a Q Learning Agent '''

    def __init__(self, name, actions, option_method=None, online=False, option_freq=1000, option_max=8, alpha=0.1, gamma=0.95,
                epsilon=0.1, default_q=1.0, option_q=1.0):

        Agent.__init__(self, name=name, actions=actions, gamma=gamma)

        self.alpha = alpha
        self.epsilon = epsilon
        self.step_number = 0
        self.default_q = default_q
        self.option_q = option_q

        self.base_actions = self.actions
        self.actions = self.actions

        self.q_func = {}

        self.option_method = option_method
        self.option_freq = 1000
        self.option_max = option_max

        self.prev_state = None
        self.prev_action = None

        self.online = online
        self.experiences = []

        self.options = []

        self.in_option = False
        self.just_finished_option = False
        self.cur_option = 0
        self.option_reward = 0
        self.option_step = 1
        self.option_start_state = None #state where option was started

        self.options_executed = 0
        self.primatives_executed = 0

    def generate_options(self, num_options):
        if len(self.options) >= self.option_max:
            return
        if self.online:
            new_ops = self.option_method(num_options, self.experiences)
        else:
            new_ops = self.option_method(num_options)

        self.options.extend(new_ops)

    def get_available_options(self, state):
        options = []
        for op in range(len(self.options)):
            if self.options[op].can_run(state):
                options.append("option-" + str(op))
        return options

    def _is_option(self, action):
        return "option" in action

    def act(self, state):
        if self.in_option:
            OPTION = self.options[self.cur_option]
            action = OPTION.act(state)
            self.step_number += 1
            self.option_step += 1
            return action

        else:
            # self.update(self.prev_state, self.prev_action, reward, state)
            action = self.epsilon_greedy_q_policy(state)
            if self._is_option(action):
                self.options_executed += 1
                self.cur_option = int(action[len("option-"):])
                self.option_reward = 0
                self.option_step = 0
                self.in_option = True
                self.just_finished_option = False
                self.option_start_state = state
                return self.act(state) #start option
            self.step_number += 1
            self.primatives_executed += 1
            return action

    def epsilon_greedy_q_policy(self, state):
        # Policy: Epsilon of the time explore, otherwise, greedyQ.
        if numpy.random.random() > self.epsilon:
            # Exploit.
            action = self.get_max_q_action(state)
        else:
            # Explore
            self.actions = self.base_actions + self.get_available_options(state)
            action = numpy.random.choice(self.actions)

        return action

    def update(self, state, action, reward, next_state, terminal, timeout):
        if self.online:
            self.experiences.append((state, action, next_state))
        #always update primatives regardless of options taken
        self.update_q_func(state, action, reward, next_state, terminal)

        if self.in_option:
            self.option_reward += reward# * self.gamma**self.option_step
            if self.options[self.cur_option].is_termination_state(next_state): #hit the termination set of the option
                self.just_finished_option = True

            if terminal or timeout: #we are done with the option but, may or may not need update / bootstrap
                self.just_finished_option = True

        if self.just_finished_option:
            #either option terminated itself, terminal reached in option, or timeout in option
            # by case: update and bootstrap, only update, no update (as if timed out before picking option)
            self.just_finished_option = False
            self.in_option = False

            if not timeout:
                OPTION = self.options[self.cur_option]
                self.update_q_func(self.option_start_state, "option-" + str(self.cur_option), self.option_reward, next_state, terminal, option_len=self.option_step)

            self.option_reward = 0
            self.option_step = 0
            self.cur_option = 0
            self.option_start_state = None
        # else:
        #     self.update_q_func(state, action, reward, next_state, terminal)


    def update_q_func(self, state, action, reward, next_state, terminal, option_len=1):
        option_len=1
        # Update the Q Function.
        max_q_curr_state = self.get_max_q_value(next_state) * (0 if terminal else 1)
        prev_q_val = self.get_q_value(state, action)
        if option_len == 1:
            self.q_func[state][action] = (1 - self.alpha) * prev_q_val + self.alpha * (reward + self.gamma*max_q_curr_state)
        else:
            self.q_func[state][action] = (1 - self.alpha) * prev_q_val + self.alpha * (reward + (self.gamma**option_len)*max_q_curr_state)


    def _compute_max_qval_action_pair(self, state):
        self.actions = self.base_actions + self.get_available_options(state)
        # Grab random initial action in case all equal
        best_action = random.choice(self.actions)
        max_q_val = float("-inf")
        shuffled_action_list = self.actions[:]
        random.shuffle(shuffled_action_list)

        for action in shuffled_action_list:
            q_s_a = self.get_q_value(state, action)
            if q_s_a > max_q_val:
                max_q_val = q_s_a
                best_action = action

        return max_q_val, best_action

    def get_max_q_action(self, state):
        return self._compute_max_qval_action_pair(state)[1]

    def get_max_q_value(self, state):
        return self._compute_max_qval_action_pair(state)[0]

    def get_value(self, state):
        return self.get_max_q_value(state)

    def get_q_value(self, state, action):
        if state in self.q_func.keys():
            if action in self.q_func[state].keys():
                return self.q_func[state][action]
            else:
                q = self.option_q if self._is_option(action) else self.default_q
                self.q_func[state][action] = q
                return q
        else:
            q = self.option_q if self._is_option(action) else self.default_q
            self.q_func[state] = {}
            self.q_func[state][action] = q
            return q

    def clear_options(self):
        self.options = []

    def reset(self):
        self.prev_state = None
        self.prev_action = None
        self.step_number = 0
        self.episode_number = 0

        self.in_option = False
        self.just_finished_option = False
        self.cur_option = 0
        self.option_reward = 0
        self.option_step = 1
        self.option_start_state = None

        self.options_executed = 0
        self.primatives_executed = 0

        self.q_func = defaultdict(lambda : defaultdict(lambda: self.default_q))
        Agent.reset(self)

    def end_of_episode(self):
        self.prev_state = None
        self.prev_action = None
        self.step_number = 0

        self.in_option = False
        self.just_finished_option = False
        self.cur_option = 0
        self.option_reward = 0
        self.option_step = 1
        self.option_start_state = None

        self.options_executed = 0
        self.primatives_executed = 0
        Agent.end_of_episode(self)


    def print_q_func(self):

        if len(self.q_func) == 0:
            print("Q Func empty!")
        else:
            for state, actiond in self.q_func.items():
                print(state)
                for action, q_val in actiond.items():
                    print("    ", action, q_val)
