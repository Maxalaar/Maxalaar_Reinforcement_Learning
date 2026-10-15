from typing import Optional
import flappy_bird_gymnasium

import gymnasium as gym
from gymnasium.envs.registration import EnvSpec


class FlappyBird(gym.Env):
    def __init__(self, environment_configuration: Optional[dict] = None):
        self.metadata = {'render_modes': ['rgb_array']}
        self.spec = EnvSpec('FlappyBird')

        self.render_mode = environment_configuration.get('render_mode', None)
        self.environment = gym.make('FlappyBird-v0', render_mode=self.render_mode, use_lidar=False)
        self.max_episode_steps = environment_configuration.get('max_episode_steps', 5_000)
        self.current_time_step = None

        self.observation_space = self.environment.observation_space
        self.action_space = self.environment.action_space

        self.observation_labels = [
            'last_pipe_horizontal_position',
            'last_top_pipe_vertical_position',
            'last_bottom_pipe_vertical_position',
            'next_pipe_horizontal_position',
            'next_top_pipe_vertical_position',
            'next_bottom_pipe_vertical_position',
            'next_next_pipe_horizontal_position',
            'next_next_top_pipe_vertical_position',
            'next_next_bottom_pipe_vertical_position',
            'player_vertical_position',
            'player_vertical_velocity',
            'player_rotation',
        ]

        self.reset()

    def reset(self, seed=None, options=None):
        self.spec.max_episode_steps = self.max_episode_steps
        self.current_time_step = 0
        observation, info = self.environment.reset(seed=seed, options=options)
        return observation, info

    def step(self, action):
        self.current_time_step += 1
        observation, reward, done, truncated, information = self.environment.step(action)

        if self.current_time_step > self.spec.max_episode_steps:
            done = True

        return observation, reward/100, done, truncated, information

    def render(self):
        return self.environment.render()

    def close(self):
        self.environment.close()
