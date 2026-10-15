import gymnasium
import numpy as np
from gymnasium.spaces import Box


def paddle_observation_space():
    observation_space = gymnasium.spaces.Dict({'position': Box(low=0, high=1, shape=(2,))})
    return observation_space


class Paddle:
    def __init__(self, environment, id):
        self.id = 'paddle_' + str(id)
        self.environment = environment
        self.speed: float = environment.paddle_speed
        self.size: float = environment.paddle_size
        self.position: np.ndarray = None

        self.reset()

    def move(self, action: int, time_step: float):
        if action == 1:
            self.position[0] -= self.speed * time_step
        elif action == 2:
            self.position[0] += self.speed * time_step
        else:
            return

        self.position = np.clip(self.position, a_min=[0, 0], a_max=[self.environment.play_area[0], self.environment.play_area[1]])

    def observation(self):
        normalize_position = np.array([self.position[0] / self.environment.play_area[0], self.position[1] / self.environment.play_area[1]])
        return {'position': normalize_position}

    def set_from_observation(self, observation):
        self.position[0] = observation[0] * self.environment.play_area[0]
        self.position[1] = observation[1] * self.environment.play_area[1]

    def reset(self):
        self.position = np.random.uniform(low=np.array([0, self.environment.play_area[1]]), high=np.array([self.environment.play_area[0], self.environment.play_area[1]]), size=2)
