import gym
from gym import error, spaces
from gym import utils
from gym.utils import seeding

from gidi_env.gidi_sim.env_single import env as sim_gidi

import logging
logger = logging.getLogger(__name__)

class GidiNew(gym.Env, utils.EzPickle):
    metadata = {'render.modes': ['human']}
    def __init__(self):
        self.env = sim_gidi()
        self.observation_space = spaces.Box(low=0., high=1.,
                                            shape=(self.env.getStateSize(),))
        self.action_space = spaces.Box(low=0., high=1., shape=(12,))

    def reset(self,testing=False):
        return self.env.reset(testing=testing)

    def step(self, action):
        ob, reward, done = self.env.step(action)
        return ob, reward, done, {}

    def render(self, mode='human', close=False):
        return

    def close(self):
        return

    def seed(self, seed=None):
        self.env.seed(seed)
