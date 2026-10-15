from typing import Optional
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import RescaleObservation, DtypeObservation
from tetris_gymnasium.envs.tetris import Tetris
from tetris_gymnasium.wrappers.observation import RgbObservation

from maxalaar_reinforcement_learning.environments.wrappers.apply_rgb_wrappers import apply_rgb_wrappers


class Tetris(gym.Env):
    def __init__(self, environment_configuration: Optional[dict] = None):
        self.render_mode = environment_configuration.get('render_mode', None)
        self.environment = gym.make('tetris_gymnasium/Tetris', render_mode=self.render_mode)

        self.metadata = self.environment.metadata
        self.metadata['render_fps'] = 15

        self.observation_in_rgb = environment_configuration.get('observation_in_rgb', False)
        if self.observation_in_rgb:
            self.environment = RgbObservation(self.environment)
            self.environment.render_scaling_factor = 10

            self.environment.observation_space.high[:] = 255
            self.environment = apply_rgb_wrappers(self.environment, environment_configuration)

            self.observation_space = self.environment.observation_space
            self.action_space = self.environment.action_space

        else:
            self.observation_space = self.environment.observation_space
            self.action_space = self.environment.action_space

    def reset(self, seed=None, options=None):
        observation, info = self.environment.reset(seed=seed, options=options)

        if self.observation_in_rgb:
            observation = observation.astype(np.float32)

        return observation, info

    def step(self, action):
        observation, reward, done, truncated, information = self.environment.step(action)

        if self.observation_in_rgb:
            observation = observation.astype(np.float32)

        return observation, reward/10, done, truncated, information

    def render(self):
        return self.environment.render()

    def close(self):
        self.environment.close()
