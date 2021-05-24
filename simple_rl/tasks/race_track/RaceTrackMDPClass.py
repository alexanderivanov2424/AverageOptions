''' GridWorldMDPClass.py: Contains the GridWorldMDP class. '''

# Python imports.
from __future__ import print_function
import random
import sys, os
import numpy as np
from collections import defaultdict
import copy

# Other imports.
from simple_rl.mdp.MDPClass import MDP
from simple_rl.tasks.race_track.RaceTrackStateClass import RaceTrackState

# Fix input to cooperate with python 2 and 3.
try:
   input = raw_input
except NameError:
   pass

class RaceTrackMDP(MDP):
    ''' Class for a Grid World MDP '''

    # Static constants.
    ACTIONS = ["up", "down", "left", "right", "neutral", "upright", "upleft", "downright", "downleft"]

    def __init__(self,
                width=5,
                height=3,
                init_loc=(1, 1),
                rand_init=False,
                rand_goal=False,
                goal_locs=[(5, 3)],
                lava_locs=[()],
                walls=[],
                is_goal_terminal=True,
                gamma=0.99,
                slip_prob=0.0,
                step_cost=0.0,
                lava_cost=0.01,
                name="racetrack"):
        '''
        Args:
            height (int)
            width (int)
            init_loc (tuple: (int, int))
            goal_locs (list of tuples: [(int, int)...])
            lava_locs (list of tuples: [(int, int)...]): These locations return -1 reward.
            walls (list)
            is_goal_terminal (bool)
        '''

        # Setup init location.
        self.rand_init = rand_init
        self.rand_goal = rand_goal

        self.init_loc = copy.deepcopy(init_loc)
        self.init_state = RaceTrackState(init_loc[0], init_loc[1], 0, 0)

        MDP.__init__(self, RaceTrackMDP.ACTIONS, self._transition_func, self._reward_func, init_state=self.init_state, gamma=gamma)

        self.init_state = RaceTrackState(init_loc[0], init_loc[1], 0, 0)

        if type(goal_locs) is not list:
            raise ValueError("(simple_rl) GridWorld Error: argument @goal_locs needs to be a list of locations. For example: [(3,3), (4,3)].")
        self.step_cost = step_cost
        self.lava_cost = lava_cost
        self.walls = walls
        self.width = width
        self.height = height
        self.goal_locs = goal_locs
        self.cur_state = RaceTrackState(init_loc[0], init_loc[1], 0, 0)
        self.is_goal_terminal = is_goal_terminal
        self.slip_prob = slip_prob
        self.name = name
        self.lava_locs = lava_locs

    def get_init_state(self):
        return self.init_state

    def get_parameters(self):
        '''
        Returns:
            (dict) key=param_name (str) --> val=param_val (object).
        '''
        param_dict = defaultdict(int)
        param_dict["width"] = self.width
        param_dict["height"] = self.height
        param_dict["init_loc"] = self.init_loc
        param_dict["rand_init"] = self.rand_init
        param_dict["goal_locs"] = self.goal_locs
        param_dict["lava_locs"] = self.lava_locs
        param_dict["walls"] = self.walls
        param_dict["is_goal_terminal"] = self.is_goal_terminal
        param_dict["gamma"] = self.gamma
        param_dict["slip_prob"] = self.slip_prob
        param_dict["step_cost"] = self.step_cost
        param_dict["lava_cost"] = self.lava_cost

        return param_dict

    def get_actions(self):
        return ["up", "down", "left", "right", "neutral", "upright", "upleft", "downright", "downleft"]

    def set_slip_prob(self, slip_prob):
        self.slip_prob = slip_prob

    def get_slip_prob(self):
        return self.slip_prob

    def is_goal_state(self, state):
        return (state.x, state.y) in self.goal_locs

    def _reward_func(self, state, action, next_state):
        '''
        Args:
            state (State)
            action (str)

        Returns
            (float)
        '''
        if (next_state.x, next_state.y) in self.goal_locs:
            return 1.0 - self.step_cost
        else:
            return 0 - self.step_cost

    def _is_goal_state_action(self, state, action):
        '''
        Args:
            state (State)
            action (str)

        Returns:
            (bool): True iff the state-action pair send the agent to the goal state.
        '''
        if (state.x, state.y) in self.goal_locs and self.is_goal_terminal:
            # Already at terminal.
            return False

        if (state.x + state.vx, state.y + state.vy) in self.goal_locs:
           return True
        else:
            return False

    def _transition_func(self, state, action):
        '''
        Args:
            state (State)
            action (str)

        Returns
            (State)
        '''

        vx = state.vx
        vy = state.vy

        newx = copy.deepcopy(state.x) + copy.deepcopy(state.vx)
        newy = copy.deepcopy(state.y) + copy.deepcopy(state.vy)

        if action == "up":
            vy += 1
        elif action == "down":
            vy -= 1
        elif action == "right":
            vx += 1
        elif action == "left":
            vx -= 1
        elif action == "upright":
            vx += 1
            vy += 1
        elif action == "upleft":
            vx -= 1
            vy += 1
        elif action == "downright":
            vx += 1
            vy -= 1
        elif action == "downleft":
            vx -= 1
            vy -= 1

        vx = min(3, max(-3, vx))
        vy = min(3, max(-3, vy))

        # print('state=', state)
        if newx > 0 and newx <= self.width and newy > 0 and newy <= self.height and not self.is_wall(newx, newy):
            next_state = RaceTrackState(newx, newy, vx, vy)
        else:
            # next_state = RaceTrackState(self.init_loc[0], self.init_loc[1], 0, 0)
            next_state = RaceTrackState(copy.deepcopy(state.x), copy.deepcopy(state.y), vx, vy)

        if (next_state.x, next_state.y) in self.goal_locs and self.is_goal_terminal:
            next_state.set_terminal(True)

        # print(next_state.x, next_state.y, self.goal_locs[0], next_state.is_terminal())
        return next_state

    def is_wall(self, x, y):
        '''
        Args:
            x (int)
            y (int)

        Returns:
            (bool): True iff (x,y) is a wall location.
        '''

        return (x, y) in self.walls

    def __str__(self):
        return self.name + "_h-" + str(self.height) + "_w-" + str(self.width)

    def __repr__(self):
        return self.__str__()

    def get_goal_locs(self):
        return self.goal_locs

    def get_lava_locs(self):
        return self.lava_locs

    def reset_init_and_goal(self):
        init_loc = random.randint(1, self.width), random.randint(1, self.height)
        while init_loc in self.walls:
            init_loc = random.randint(1, self.width), random.randint(1, self.height)

        self.cur_state = RaceTrackState(init_loc[0], init_loc[1], 0, 0)
        self.init_state = RaceTrackState(init_loc[0], init_loc[1], 0, 0)
        self.init_loc = init_loc

        self.goal_locs = [(random.randint(1, self.width), random.randint(1, self.height))]
        while self.goal_locs[0] in self.walls:
            self.goal_locs = [(random.randint(1, self.width), random.randint(1, self.height))]

    def reset_goal(self):
        self.goal_locs = [(random.randint(1, self.width), random.randint(1, self.height))]
        while self.goal_locs[0] in self.walls:
            self.goal_locs = [(random.randint(1, self.width), random.randint(1, self.height))]

    def reset(self):
        # if self.rand_init:
        #     init_loc = random.randint(1, self.width), random.randint(1, self.height)
        #     self.cur_state = RaceTrackState(init_loc[0], init_loc[1], 0, 0)
        #     self.init_loc = init_loc
        # else:
        # self.cur_state = copy.deepcopy(self.init_state)
        self.cur_state = RaceTrackState(self.init_loc[0], self.init_loc[1], 0, 0)



def make_race_track_from_file(file_name, randomize=False, rand_init_and_goal=False, num_goals=1, name=None, goal_num=None, slip_prob=0.0, step_cost=0.0):
    '''
    Args:
        file_name (str)
        randomize (bool): If true, chooses a random agent location and goal location.
        num_goals (int)
        name (str)

    Returns:
        (RaceTrackMDP)

    Summary:
        Builds a GridWorldMDP from a file:
            'w' --> wall
            'a' --> agent
            'g' --> goal
            '-' --> empty
    '''

    if name is None:
        name = file_name.split(".")[0]

    # grid_path = os.path.dirname(os.path.realpath(__file__))
    wall_file = open(os.path.join(os.getcwd(), file_name))
    wall_lines = wall_file.readlines()

    # Get walls, agent, goal loc.
    num_rows = len(wall_lines)
    num_cols = len(wall_lines[0].strip())
    agent_x, agent_y = 1, 1
    walls = []
    goal_locs = []
    lava_locs = []

    for i, line in enumerate(wall_lines):
        line = line.strip()
        for j, ch in enumerate(line):
            if ch == "w":
                walls.append((j + 1, num_rows - i))
            elif ch == "g":
                goal_locs.append((j + 1, num_rows - i))
            elif ch == "l":
                lava_locs.append((j + 1, num_rows - i))
            elif ch == "a":
                agent_x, agent_y = j + 1, num_rows - i
            elif ch == "-":
                pass
                # empty_cells.append((j + 1, num_rows - i))

    if goal_num is not None:
        goal_locs = [goal_locs[goal_num % len(goal_locs)]]

    if len(goal_locs) == 0:
        goal_locs = [(num_cols, num_rows)]

    return RaceTrackMDP(width=num_cols, height=num_rows, init_loc=(agent_x, agent_y), goal_locs=goal_locs, rand_init=rand_init_and_goal, rand_goal=rand_init_and_goal, lava_locs=lava_locs, walls=walls, name=name, step_cost=step_cost, slip_prob=slip_prob)

    # def reset(self):
    #     if self.rand_init:
    #         init_loc = random.randint(1, width), random.randint(1, height)
    #         self.cur_state = RaceTrackState(init_loc[0], init_loc[1])
    #     else:
    #         self.cur_state = copy.deepcopy(self.init_state)

def main():
    grid_world = RaceTrackMDP(5, 10, (1, 1), (6, 7))

    pass

if __name__ == "__main__":
    main()
