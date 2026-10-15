from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box


class BipedalWalker(gym.Env):
    def __init__(self, environment_configuration: Optional[dict] = None):
        self.metadata = {'render_modes': ['rgb_array']}

        self.render_mode = environment_configuration.get('render_mode', None)
        self.hardcore = environment_configuration.get('hardcore', False)
        self.environment = gym.make('BipedalWalker-v3', render_mode=self.render_mode, hardcore=self.hardcore)

        self.observation_space = Box(
            low=np.NINF,
            high=np.PINF,
            shape=(24,),
            dtype=np.float32
        )
        self.action_space = self.environment.action_space

        self.observation_labels = [
            # Hull
            'hull_angle_speed',
            'hull_angular_velocity',
            'hull_horizontal_speed',
            'hull_vertical_speed',
            # leg 0
            'joint_0_angle',
            'joint_0_speed',
            'joint_1_angle',
            'joint_1_speed',
            'leg_0_contact_ground',
            # leg 1
            'joint_2_angle',
            'joint_2_speed',
            'joint_3_angle',
            'joint_3_speed',
            'leg_1_contact_ground',
            # Lidar
            'lidar_0',
            'lidar_1',
            'lidar_2',
            'lidar_3',
            'lidar_4',
            'lidar_5',
            'lidar_6',
            'lidar_7',
            'lidar_8',
            'lidar_9',
        ]

    def reset(self, seed=None, options=None):
        observation, info = self.environment.reset(seed=seed, options=options)
        return observation, info

    def step(self, action):
        observation, reward, done, truncated, information = self.environment.step(action)

        return observation, reward/30, done, truncated, information

    def render(self):
        return self.environment.render()

    def close(self):
        self.environment.close()
