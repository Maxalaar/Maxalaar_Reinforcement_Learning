import os
from typing import Optional
import gymnasium as gym
import gymnasium.spaces
from gymnasium.wrappers import TransformReward, RescaleAction
from gymnasium import spaces
from gymnasium.spaces import Box, Dict
import numpy as np

VizdoomEnv = None
gymnasium_wrapper = None
try:
    from vizdoom.gymnasium_wrapper.base_gymnasium_env import VizdoomEnv
    from vizdoom import gymnasium_wrapper
except ImportError:
    print('ViZDoom is not installed.')

# Assume this exists in the same package
from maxalaar_reinforcement_learning.environments.wrappers.apply_rgb_wrappers import apply_rgb_wrappers


class ClipContinuousActionWrapper(gym.ActionWrapper):
    def __init__(self, environment, low, high, keys_to_clip=('continuous',)):
        super().__init__(environment)
        self.keys_to_clip = keys_to_clip
        self.low = low
        self.high = high

        assert isinstance(environment.action_space, Dict), "Action space must be a Dict"

        new_action_space = {}
        for key, space in environment.action_space.items():
            if key in keys_to_clip:
                new_action_space[key] = Box(
                    low=self.low,
                    high=self.high,
                    shape=space.shape,
                    dtype=space.dtype
                )
            else:
                new_action_space[key] = space

        self.action_space = gymnasium.spaces.Dict(new_action_space)

    def action(self, action):
        for key in self.keys_to_clip:
            if key in action:
                action[key] = np.clip(action[key], self.low, self.high)
        return action


class ScreenOnlyWrapper(gym.ObservationWrapper):
    def __init__(self, environment):
        super().__init__(environment)
        observation_space = environment.observation_space
        # Add safety check for Dict space
        if not isinstance(observation_space, spaces.Dict):
            raise ValueError("ScreenOnlyWrapper requires Dict observation space")

        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=observation_space['screen'].shape,
            dtype=np.uint8
        )

    def observation(self, observation):
        return observation['screen']


class ViZDoom(gym.Env):
    def __init__(self, environment_configuration: dict):
        super().__init__()

        # Validate required parameters
        if 'game_name' not in environment_configuration:
            raise ValueError("environment_configuration must include 'game_name'")

        self.metadata = {
            'render_modes': ['rgb_array'],
            'render_fps': environment_configuration.get('render_fps', 60) / environment_configuration.get('frame_skip', 4),
        }
        self.render_mode = environment_configuration.get('render_mode')
        self.game_name = environment_configuration['game_name']

        # Setup temporary directory
        temp_dir = os.path.abspath("./temporary/vizdoom")
        os.makedirs(temp_dir, exist_ok=True)
        original_dir = os.getcwd()

        environment = None
        try:
            os.chdir(temp_dir)

            # Create base environment with environment_configuration parameters
            # environment = gym.make(
            #     self.game_name,
            #     frame_skip=environment_configuration.get('frame_skip', 4),
            #     render_mode=self.render_mode,
            #     use_multi_binary_action_space=True,
            #     **environment_configuration.get('gym_make_kwargs', {})
            # )
            environment = VizdoomEnv(
                level='deathmatch',
                frame_skip=environment_configuration.get('frame_skip', 4),
                render_mode=self.render_mode,
                **environment_configuration.get('gym_make_kwargs', {})
            )

            # Apply wrappers
            environment = ScreenOnlyWrapper(environment)
            environment = apply_rgb_wrappers(environment, environment_configuration)

            if environment_configuration.get('clip_continuous_action', None) is not None:
                environment = ClipContinuousActionWrapper(environment, low=-environment_configuration.get('clip_continuous_action', None), high=environment_configuration.get('clip_continuous_action', None))

            # Reward scaling
            reward_scale = environment_configuration.get('reward_scale_factor', 1.0)
            environment = TransformReward(environment, lambda r: reward_scale * r)

            # Assign successfully created environment
            self.environment = environment
            environment = None  # Prevent cleanup after successful assignment

            # Set spaces
            self.observation_space = self.environment.observation_space
            self.action_space = self.environment.action_space

        finally:
            # Always revert directory and clean up on failure
            os.chdir(original_dir)
            if environment is not None:
                environment.close()

    def reset(self, seed=None, options=None):
        return self.environment.reset(seed=seed, options=options)

    def step(self, action):
        return self.environment.step(action)

    def render(self):
        return self.environment.render()

    def close(self):
        if hasattr(self, 'environment') and self.environment is not None:
            self.environment.close()
            # Prevent double-close
            self.environment = None