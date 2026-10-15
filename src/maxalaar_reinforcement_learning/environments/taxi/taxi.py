from typing import Optional
import gymnasium as gym
from gymnasium.spaces import Dict, Discrete

def taxi_decompose_observation(observation: int) -> dict:
    """
    Decompose an encoded observation into its components.

    Args:
        observation (int): Encoded observation.

    Returns:
        dict: A dictionary with components `taxi_row`, `taxi_column`, `passenger_location`, and `destination`.
    """
    # Decompose observation
    destination = observation % 4
    observation //= 4
    passenger_location = observation % 5
    observation //= 5
    taxi_column = observation % 5
    taxi_row = observation // 5

    # Return the decomposed components
    return {
        'taxi_row': taxi_row,
        'taxi_column': taxi_column,
        'passenger_location': passenger_location,
        'destination': destination,
    }


class Taxi(gym.Env):
    def __init__(self, environment_configuration: Optional[dict] = None):
        self.metadata = {
            'render_modes': ['rgb_array'],
            'render_fps': 5,
        }

        self.render_mode = environment_configuration.get('render_mode', None)
        self.environment = gym.make('Taxi-v3', render_mode=self.render_mode)

        self.use_new_observation_space = environment_configuration.get('new_observation_space', False)
        if self.use_new_observation_space:
            self.observation_space = Dict({
                'taxi_row': Discrete(5),
                'taxi_column': Discrete(5),
                'passenger_location': Discrete(5),
                'destination': Discrete(4),
            })
            self.observation_labels = [
                'taxi_row',
                'taxi_column',
                'passenger_location',
                'destination',
            ]
        else:
            self.observation_space = self.environment.observation_space

        self.action_space = self.environment.action_space

    def reset(self, seed=None, options=None):
        observation, information = self.environment.reset(seed=seed, options=options)

        if self.use_new_observation_space:
            observation = taxi_decompose_observation(observation)

        return observation, information

    def step(self, action):
        observation, reward, done, truncated, information = self.environment.step(action)

        if self.use_new_observation_space:
            observation = taxi_decompose_observation(observation)

        return observation, reward, done, truncated, information

    def render(self, mode='human'):
        return self.environment.render()

    def close(self):
        self.environment.close()
