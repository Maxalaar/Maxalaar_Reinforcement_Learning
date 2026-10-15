from typing import Optional
import gymnasium as gym
import ale_py
import numpy as np
import cv2  # Import OpenCV
import time  # Pour contrôler le FPS
from gymnasium.spaces import Box
from gymnasium.wrappers import TransformReward
from ray.rllib.env.wrappers.atari_wrappers import wrap_atari_for_new_api_stack, EpisodicLifeEnv, ClipRewardEnv, NoopResetEnv, WarpFrame

from maxalaar_reinforcement_learning.environments.wrappers.apply_rgb_wrappers import apply_rgb_wrappers

gym.register_envs(ale_py)


class Atari(gym.Env):
    def __init__(self, environment_configuration: Optional[dict] = None):
        self.metadata = {
            'render_modes': ['rgb_array'],
            'render_fps': environment_configuration.get('render_fps', 60 / environment_configuration.get('frameskip', 1)),
        }

        self.render_mode = environment_configuration.get('render_mode', None)
        self.game_name: str = environment_configuration.get('game_name')

        self.environment = gym.make(
            'ALE/' + self.game_name,
            difficulty=environment_configuration.get('difficulty', 0),
            render_mode=self.render_mode,
            repeat_action_probability=environment_configuration.get('repeat_action_probability', 0.25),
            frameskip=environment_configuration.get('frameskip', 1),
            full_action_space=environment_configuration.get('full_action_space', False),
            obs_type='rgb',
        )

        self.environment = apply_rgb_wrappers(self.environment, environment_configuration)
        if environment_configuration.get('use_life_equal_episode', False):
            self.environment = EpisodicLifeEnv(self.environment)
        self.environment = TransformReward(self.environment, lambda r: environment_configuration.get('reward_scale_factor', 1.0) * r)

        # self.environment = wrap_atari_for_new_api_stack(
        #     self.environment,
        #     framestack=environment_configuration.get('frame_stack_size', 4),
        #     dim=environment_configuration.get('resize_observation_shape', 64),
        #     frameskip=-1,
        # )
        # self.environment = TransformReward(self.environment, lambda r: r / 100)

        self.observation_space = self.environment.observation_space
        self.action_space = self.environment.action_space

    def reset(self, seed=None, options=None):
        observation, information = self.environment.reset(seed=seed, options=options)
        return observation, information

    def step(self, action):
        observation, reward, done, truncated, information = self.environment.step(action)
        return observation, reward, done, truncated, information

    def render(self):
        return self.environment.render()

    def close(self):
        self.environment.close()
