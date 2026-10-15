from typing import Optional

import gymnasium as gym
from gymnasium.spaces import Box
import numpy as np


class LunarLander(gym.Env):
    def __init__(self, environment_configuration: Optional[dict] = None):
        self.metadata = {'render_modes': ['rgb_array']}

        self.render_mode = environment_configuration.get('render_mode', None)
        self.environment = gym.make('LunarLander-v3', render_mode=self.render_mode)

        self.observation_space= Box(
            low=np.NINF,
            high=np.PINF,
            shape=(8,),
            dtype=np.float32
        )
        self.action_space = self.environment.action_space

        self.observation_labels = [
            'lander_coordinate_x',
            'lander_coordinate_y',
            'linear_velocity_x',
            'linear_velocity_y',
            'angle',
            'angular_velocity',
            'leg_0_contact_ground',
            'leg_1_contact_ground',
        ]
        self.action_labels = [
            'do_nothing',
            'fire_left_orientation_engine',
            'fire_main_engine',
            'fire_right_orientation_engine',
        ]

    def reset(self, seed=None, options=None):
        observation, info = self.environment.reset(seed=seed, options=options)
        return observation, info

    def step(self, action):
        observation, reward, done, truncated, information = self.environment.step(action)

        return observation, reward/100, done, truncated, information

    def render(self, mode='human'):
        return self.environment.render()

    def close(self):
        self.environment.close()
