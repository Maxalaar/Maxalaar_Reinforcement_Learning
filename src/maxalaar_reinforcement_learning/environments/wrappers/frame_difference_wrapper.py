import numpy as np
from gymnasium import ObservationWrapper
from gymnasium.spaces import Box


class FrameDifferenceWrapper(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.orig_space = env.observation_space

        # Déterminer le type de données pour les différences
        if self.orig_space.dtype == np.uint8:
            self.diff_dtype = np.int16  # Pour éviter les problèmes de signe
        else:
            self.diff_dtype = self.orig_space.dtype

        orig_shape = self.orig_space.shape
        c, h, w = orig_shape  # Forme (canal, hauteur, largeur)

        # Nouvelle forme : (2 * nb_canaux, hauteur, largeur)
        new_shape = (2 * c, h, w)

        # Créer des bornes infinies pour la partie différence
        low_orig = self.orig_space.low
        high_orig = self.orig_space.high

        # Utiliser -inf et +inf pour les différences
        low_diff = np.full_like(low_orig, -np.inf)
        high_diff = np.full_like(high_orig, np.inf)

        # Concaténation sur l'axe des canaux (axe 0)
        low = np.concatenate([low_diff, low_diff], axis=0)
        high = np.concatenate([high_diff, high_diff], axis=0)

        self.observation_space = Box(
            low=low,
            high=high,
            dtype=self.diff_dtype,
            shape=new_shape
        )
        self.last_obs = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # Convertir et stocker la première observation
        self.last_obs = obs.astype(self.diff_dtype)
        # Différence nulle pour la première frame
        diff = np.zeros_like(self.last_obs)
        # Concaténation sur l'axe des canaux (0)
        combined = np.concatenate([self.last_obs, diff], axis=0)
        return combined, info

    def observation(self, obs):
        # Conversion au type approprié
        current_obs = obs.astype(self.diff_dtype)
        # Calcul de la différence
        diff = current_obs - self.last_obs
        # Mise à jour de la dernière observation
        self.last_obs = current_obs

        # Concaténation sur l'axe des canaux (0)
        combined = np.concatenate([current_obs, diff], axis=0)
        return combined