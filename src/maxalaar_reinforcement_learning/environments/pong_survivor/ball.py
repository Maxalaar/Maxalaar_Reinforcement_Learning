import gymnasium
import numpy as np
from gymnasium.spaces import MultiDiscrete, Box


def ball_observation_space():
    observation_space = gymnasium.spaces.Dict({'position': Box(low=0, high=1, shape=(2,)), 'velocity': Box(low=-1, high=1, shape=(2,))})
    return observation_space


class Ball:
    def __init__(self, environment, id):
        self.matriculation = id
        self.id = 'ball_' + str(id)
        self.environment = environment
        self.speed: float = environment.ball_speed
        self.position: np.ndarray = None
        self.angle: float = None
        self.velocity: np.ndarray = None

        self.reset()

    def _compute_velocity(self):
        x = np.cos(self.angle) * self.speed
        y = np.sin(self.angle) * self.speed
        return np.array([x, y])

    def move(self, time_step):
        self.position += self.velocity * time_step

        self.position = np.clip(self.position, a_min=[0, 0], a_max=[self.environment.play_area[0], self.environment.play_area[1]])

        if self.position[0] <= 0.0 or self.position[0] >= self.environment.play_area[0]:
            self.velocity[0] *= -1
        if self.position[1] <= 0.0 or self.position[1] >= self.environment.play_area[1]:
            self.velocity[1] *= -1

        if self.position[1] >= self.environment.play_area[1]:
            for paddle in self.environment.paddles:
                if paddle.position[0] - paddle.size/2 < self.position[0] < paddle.position[0] + paddle.size/2:
                    return

            self.environment.terminated = True

    def observation(self):
        normalize_position = np.array([self.position[0]/self.environment.play_area[0], self.position[1]/self.environment.play_area[1]])
        normalize_velocity = self.velocity / self.speed
        return {'position': normalize_position, 'velocity': normalize_velocity}

    def set_from_observation(self, observation):
        self.position[0] = observation[0] * self.environment.play_area[0]
        self.position[1] = observation[1] * self.environment.play_area[1]
        self.velocity = observation[2:] * self.speed

    def reset(self):
        self.position = np.random.uniform(low=np.array([0, 0]), high=self.environment.play_area, size=2)
        self.angle = np.random.uniform(0, 2*np.pi)
        self.velocity = self._compute_velocity()



