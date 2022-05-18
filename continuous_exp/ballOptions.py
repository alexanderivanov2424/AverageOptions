import numpy as np
from continuous_exp.utils import inRange, setEnvVec, vecToPos




class Option:

    def __init__(self, init_vec, term_vec, radius):
        self.init_vec = init_vec
        self.radius = radius
        self.term_vec = term_vec

    def canRun(self, env, vec, use_pos=False, idx=None):
        v1 = vecToPos(env, vec, use_pos=use_pos, idx=idx)
        v2 = vecToPos(env, self.init_vec, use_pos=use_pos, idx=idx)
        return inRange(v1, v2, self.radius)

    def run(self, env):
        setEnvVec(env, self.term_vec)