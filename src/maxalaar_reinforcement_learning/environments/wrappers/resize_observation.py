import gymnasium as gym
import numpy as np
from gymnasium import spaces
import cv2


class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super().__init__(env)
        self.shape = tuple(shape)

        # Vérifier si l'observation est une image
        if len(self.observation_space.shape) == 3:
            self.observation_space = spaces.Box(
                low=0,
                high=255,
                shape=(self.shape[0], self.shape[1], self.observation_space.shape[2]),
                dtype=np.uint8
            )
        elif len(self.observation_space.shape) == 2:
            self.observation_space = spaces.Box(
                low=0,
                high=255,
                shape=self.shape,
                dtype=np.uint8
            )
        else:
            raise NotImplementedError("Unsupported observation shape")

    def observation(self, observation):
        # Redimensionner l'observation
        resized_obs = cv2.resize(
            observation,
            self.shape[::-1],  # OpenCV attend (largeur, hauteur)
            interpolation=cv2.INTER_AREA
        )

        # Maintenir le nombre de canaux si l'image avait des couleurs
        if len(observation.shape) == 3 and observation.shape[2] > 1:
            return resized_obs
        else:
            return resized_obs.reshape(self.shape)
