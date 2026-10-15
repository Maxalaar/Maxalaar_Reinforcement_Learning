"""
Port of Neural Slime Volleyball to Python Gymnasium Environment

David Ha (2020)

Original version:

https://otoro.net/slimevolley
https://blog.otoro.net/2015/03/28/neural-slime-volleyball/
https://github.com/hardmaru/neuralslimevolley

No dependencies apart from Numpy and Gymnasium
"""

import logging
import math
from typing import Optional, Dict

import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
from gymnasium.envs.registration import register
import numpy as np
import pygame
from pygame.locals import *
from collections import deque

from maxalaar_reinforcement_learning.environments.slime_volley.baseline_policy import BaselinePolicy
from maxalaar_reinforcement_learning.environments.slime_volley.game import Game

np.set_printoptions(threshold=20, precision=3, suppress=True, linewidth=200)

class SlimeVolley(gym.Env):
    def __init__(self, environment_configuration: Optional[dict] = {}):
        # Game settings
        self.REF_W = 24 * 2
        self.REF_H = self.REF_W
        self.REF_U = 1.5  # ground height
        self.REF_WALL_WIDTH = 1.0  # wall width
        self.REF_WALL_HEIGHT = 3.5
        self.PLAYER_SPEED_X = 10 * 1.75
        self.PLAYER_SPEED_Y = 10 * 1.35
        self.MAX_BALL_SPEED = 15 * 1.5
        self.TIMESTEP = environment_configuration.get('time_step', 1 / 30.)
        self.NUDGE = 0.1
        self.FRICTION = 1.0  # 1 means no FRICTION, less means FRICTION
        self.INIT_DELAY_FRAMES = 30
        self.GRAVITY = -9.8 * 2 * 1.5
        self.MAXLIVES = 5  # game ends when one agent loses this many games
        self.WINDOW_WIDTH = 1200
        self.WINDOW_HEIGHT = 500
        self.FACTOR = self.WINDOW_WIDTH / self.REF_W

        # Colors
        self.BALL_COLOR = (217, 79, 0)
        self.AGENT_LEFT_COLOR = (35, 93, 188)
        self.AGENT_RIGHT_COLOR = (255, 236, 0)
        self.BACKGROUND_COLOR = (11, 16, 19)
        self.FENCE_COLOR = (102, 56, 35)
        self.COIN_COLOR = self.FENCE_COLOR
        self.GROUND_COLOR = (116, 114, 117)

        self.metadata = {
            'render_modes': ['human', 'rgb_array'],
            'render_fps': 50,
        }
        self.render_mode = environment_configuration.get('render_mode', 'rgb_array')

        self.t = 0
        self.t_limit = 4000

        self.action_space = spaces.Box(low=-1, high=1, shape=(3,))
        # self.action_space = spaces.MultiDiscrete(np.array([2, 2, 2]))

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32)
        self.game = Game(self)
        self.viewer = None
        self.screen = None
        self.policy = BaselinePolicy()
        self.otherAction = None

        self.observation_labels = [
            # Self
            'self_position_x',
            'self_position_y',
            'self_velocity_x',
            'self_velocity_y',
            # Ball
            'ball_position_x',
            'ball_position_y',
            'ball_velocity_x',
            'ball_velocity_y',
            # Other
            'other_position_x',
            'other_position_y',
            'other_velocity_x',
            'other_velocity_y',
        ]

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        self.game = Game(self, np_random=self.np_random)
        return [seed]

    def getObs(self):
        return self.game.agent_left.getObservation()

    def step(self, action, otherAction=None):
        done = False
        self.t += 1

        if self.otherAction is not None:
            otherAction = self.otherAction

        if otherAction is None:
            obs = self.game.agent_right.getObservation()
            otherAction = self.policy.predict(obs)

        self.game.agent_left.setAction(action)
        self.game.agent_right.setAction(otherAction)

        reward = self.game.step()

        obs = self.getObs()

        if self.t >= self.t_limit:
            done = True

        if self.game.agent_left.life <= 0 or self.game.agent_right.life <= 0:
            done = True

        info = {
            'left_agent_lives': self.game.agent_left.lives(),
            'right_agent_lives': self.game.agent_right.lives(),
            'left_agent_observation': self.game.agent_left.getObservation(),
            'right_agent_observation': self.game.agent_right.getObservation(),
        }

        return obs, reward, done, False, info

    def reset(self, seed=None, options=None):
        self.t = 0
        self.game.reset()
        return self.getObs(), {}

    def render(self):
        if self.render_mode == 'human':
            if self.screen is None:
                pygame.init()
                self.screen = pygame.display.set_mode((self.WINDOW_WIDTH, self.WINDOW_HEIGHT))
                pygame.display.set_caption('Slime Volleyball')
            surface = pygame.Surface((self.WINDOW_WIDTH, self.WINDOW_HEIGHT))
            self.game.display(surface)
            self.screen.blit(surface, (0, 0))
            pygame.display.flip()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
        elif self.render_mode == 'rgb_array':
            surface = pygame.Surface((self.WINDOW_WIDTH, self.WINDOW_HEIGHT))
            self.game.display(surface)
            return np.transpose(pygame.surfarray.array3d(surface), (1, 0, 2))
        return None

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None