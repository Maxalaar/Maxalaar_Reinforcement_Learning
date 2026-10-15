from typing import Optional

import gymnasium as gym
from gymnasium.spaces import Box
import numpy as np
from gymnasium.wrappers import ResizeObservation, GrayscaleObservation, FrameStackObservation, DtypeObservation, \
    RescaleObservation


class CarRacing(gym.Env):
    def __init__(self, environment_configuration: Optional[dict] = None):
        self.metadata = {'render_modes': ['rgb_array']}

        self.render_mode = environment_configuration.get('render_mode', None)
        self.environment = gym.make('CarRacing-v3', render_mode=self.render_mode)
        self.environment = ResizeObservation(self.environment, (64, 64))
        self.environment = GrayscaleObservation(self.environment, keep_dim=False)
        self.environment = FrameStackObservation(self.environment, stack_size=4)
        self.environment = DtypeObservation(self.environment, dtype=np.float32)
        self.environment = RescaleObservation(
            self.environment,
            min_obs=np.full(self.environment.observation_space.shape, fill_value=-1, dtype=np.float32),
            max_obs=np.full(self.environment.observation_space.shape, fill_value=1, dtype=np.float32),
        )

        self.observation_space = self.environment.observation_space
        self.action_space = self.environment.action_space

        # self.observation_space = gym.spaces.Box(low=0, high=255, shape=(3, 96, 96), dtype=np.float32)
        # self.action_space = self.environment.action_space

    def reset(self, seed=None, options=None):
        observation, info = self.environment.reset(seed=seed, options=options)
        # observation = np.transpose(observation, (2, 0, 1)).astype(np.float32)
        return observation, info

    def step(self, action):
        observation, reward, done, truncated, information = self.environment.step(action)
        # observation = np.transpose(observation, (2, 0, 1)).astype(np.float32)
        return observation, reward / 100, done, truncated, information

    def render(self, mode='human'):
        return self.environment.render()

    def close(self):
        self.environment.close()
