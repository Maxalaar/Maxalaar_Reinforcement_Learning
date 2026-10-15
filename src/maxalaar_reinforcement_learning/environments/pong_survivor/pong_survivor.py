import gymnasium as gym
import gymnasium.spaces
import numpy as np

from typing import TYPE_CHECKING, Any, SupportsFloat, TypeVar, Optional

from gymnasium.envs.registration import EnvSpec
from gymnasium.spaces import Box

from maxalaar_reinforcement_learning.environments.pong_survivor.ball import Ball, ball_observation_space
from maxalaar_reinforcement_learning.environments.pong_survivor.paddle import Paddle, paddle_observation_space
from maxalaar_reinforcement_learning.environments.pong_survivor.render import RenderEnvironment

if TYPE_CHECKING:
    pass

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")
RenderFrame = TypeVar("RenderFrame")


def flattening(dictionary: dict, flatten_dictionary: dict = {}, prefix: str = ''):
    for key, value in dictionary.items():
        if type(value) is dict:
            flattening(
                dictionary=value,
                flatten_dictionary=flatten_dictionary,
                prefix=prefix+'_'+str(key)
            )
        else:
            if value is not None:
                flatten_dictionary[prefix+'_'+str(key)] = value

    return flatten_dictionary


class PongSurvivor(gym.Env):
    def __init__(self, environment_configuration: Optional[dict] = {}):
        self.metadata = {
            'render_modes': ['rgb_array'],
            'render_fps': 20,
        }

        self.balls = []
        self.paddles = []

        self.paddle_size = environment_configuration.get('paddle_size', 30)
        self.paddle_speed = environment_configuration.get('paddle_speed', 40.0)
        self.ball_speed = environment_configuration.get('ball_speed', 20.0)

        self.time_step: float = 0.02
        self.max_time: float = environment_configuration.get('max_time', 50)
        self.spec = EnvSpec('PongSurvivor')
        self.frame_skip = environment_configuration.get('frame_skip', 0)
        self.spec.max_episode_steps = int(self.max_time / (self.time_step * (self.frame_skip + 1)))

        self.play_area: np.ndarray = np.array(
            [environment_configuration.get('size_map_x', 100), environment_configuration.get('size_map_y', 100)])

        for i in range(environment_configuration.get('number_ball', 1)):
            self.balls.append(Ball(self, i))
        for i in range(environment_configuration.get('number_paddle', 1)):
            self.paddles.append(Paddle(self, i))

        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = self._get_observation_space()

        self.render_mode = environment_configuration.get('render_mode', 'rgb_array')
        self.render_environment = None
        self.display_arrows = environment_configuration.get('display_arrows', True)
        self.arrow_size = environment_configuration.get('display_arrows', 50)
        self.display_number = environment_configuration.get('display_number', True)

        self._current_time_steps: int = None
        self.terminated = None
        self.truncated = None

        self.reset()
        self.observation_labels = self._get_observation_labels()
        self.action_labels = self._get_action_labels()

    def reset(self, *, seed=None, options=None):
        self.spec.max_episode_steps = int(self.max_time / (self.time_step * (self.frame_skip + 1)))
        self._current_time_steps = 0

        self.terminated = False
        self.truncated = False

        for ball in self.balls:
            ball.reset()

        for paddle in self.paddles:
            paddle.reset()

        return self._get_observation(), self._get_information()

    def step(self, action):
        self._current_time_steps += 1

        frame_current_number = 0
        while not self.terminated and not self.truncated and frame_current_number <= self.frame_skip:
            frame_current_number += 1
            for ball in self.balls:
                ball.move(self.time_step)

            self.paddles[0].move(action, self.time_step)

            if self._current_time_steps > self.spec.max_episode_steps:
                self.terminated = True

        return self._get_observation(), 1.0 / (self.spec.max_episode_steps + 1), self.terminated, self.truncated, self._get_information()

    def render(self):
        if self.render_mode is not None:
            if self.render_environment is None:
                self.render_environment = RenderEnvironment(self)

            rendering = self.render_environment.render()
            return rendering

    def _get_observation_space(self):
        observation_space = {}

        for ball in self.balls:
            observation_space[ball.id] = ball_observation_space()

        for paddle in self.paddles:
            observation_space[paddle.id] = paddle_observation_space()

        observation_space['time_percentage'] = Box(low=-2, high=2, shape=(1,))

        return gymnasium.spaces.Dict(observation_space)

    def _get_observation_labels(self):
        observation_labels = []

        for ball in self.balls:
            observation_labels.append(str(ball.id) + '_position_x')
            observation_labels.append(str(ball.id) + '_position_y')
            observation_labels.append(str(ball.id) + '_velocity_x')
            observation_labels.append(str(ball.id) + '_velocity_y')

        for paddle in self.paddles:
            observation_labels.append(str(paddle.id) + '_position_x')
            observation_labels.append(str(paddle.id) + '_position_y')

        observation_labels.append('time_percentage')
        return observation_labels

    def _get_action_labels(self):
        return ['idle', 'left', 'right']

    def _get_observation(self):
        observation = {}

        for ball in self.balls:
            observation[ball.id] = ball.observation()

        for paddle in self.paddles:
            observation[paddle.id] = paddle.observation()

        observation['time_percentage'] = np.array([float(self._current_time_steps) / float(self.spec.max_episode_steps)])

        return observation

    def set_from_observation(self, observation):
        index = 0

        for ball in self.balls:
            ball.set_from_observation(observation[index:index+4])
            index += 4

        for paddle in self.paddles:
            paddle.set_from_observation(observation[index:index+2])
            index += 2

    def _get_information(self):
        information = {}
        information.update({
            'observation': self._get_observation(),
        })
        return flattening(information)
