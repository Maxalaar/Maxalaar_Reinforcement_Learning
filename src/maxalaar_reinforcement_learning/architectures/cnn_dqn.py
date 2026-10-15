# from typing import Dict, Any
#
# import torch
# import torch.nn as nn
# from ray.rllib.core import Columns
# from ray.rllib.core.rl_module.torch import TorchRLModule
#
#
# class CNNDQN(TorchRLModule):
#     def __init__(self, observation_space, action_space, inference_only, model_config, catalog_class):
#         """
#         observation_space: espace d'observation, dont .shape renvoie (4, 84, 84)
#         action_space: espace d'action (discret) avec action_space.number_datas donnant le nombre d'actions
#         config: dictionnaire de configuration (non utilisé ici, mais potentiellement utile pour paramétrer l'architecture)
#         """
#         super().__init__()  # observation_space, action_space, model_config
#
#         # Définition du bloc CNN inspiré de l'architecture DQN classique (Mnih et al., 2015)
#         self.cnn = nn.Sequential(
#             nn.Conv2d(in_channels=observation_space.shape[0], out_channels=32, kernel_size=8, stride=4),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
#             nn.ReLU(),
#             nn.Flatten()
#         )
#
#         # Calcul de la taille de sortie du CNN pour configurer la première couche du réseau fully connected.
#         with torch.no_grad():
#             dummy_input = torch.zeros(1, *observation_space.shape)
#             cnn_output = self.cnn(dummy_input)
#             cnn_output_size = cnn_output.shape[1]
#
#         # Réseau entièrement connecté pour transformer les features en Q-valeurs
#         self.fc = nn.Sequential(
#             nn.Linear(cnn_output_size, 512),
#             nn.ReLU(),
#             nn.Linear(512, action_space.number_datas)
#         )
#
#     def forward_target(self, batch: Dict[str, Any]) -> Dict[str, Any]:
#         # return {'af': self.critic_layers(batch[Columns.OBS])}
#         features = self.cnn(batch[Columns.OBS])
#         q_values = self.fc(features)
#         return q_values
#
#     # def forward(self, x, **kwargs):
#     #     """
#     #     La méthode forward prend en entrée un tenseur de forme [batch, 4, 84, 84]
#     #     et retourne les Q-valeurs pour chaque action.
#     #     """
#     #     features = self.cnn(x)
#     #     q_values = self.fc(features)
#     #     return q_values
#     #
#     # def forward_inference(self, x, **kwargs):
#     #     """
#     #     Méthode appelée lors de l'inférence (évaluation). Ici, on réutilise forward.
#     #     """
#     #     return self.forward(x, **kwargs)
#     #
#     # def forward_train(self, x, **kwargs):
#     #     """
#     #     Méthode appelée lors de l'entraînement. Pour DQN, on peut également utiliser la même logique.
#     #     """
#     #     return self.forward(x, **kwargs)

import torch
import torch.nn as nn
from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.core.rl_module.apis import ValueFunctionAPI, TargetNetworkAPI
from ray.rllib.utils.annotations import override


class CNNDQN(TorchRLModule, TargetNetworkAPI):
    @override(TorchRLModule)
    def setup(self):
        # Récupération de la fonction d'activation et de la configuration des couches denses
        self.activation_function = self.model_config.get('activation_function', nn.ReLU())
        self.configuration_hidden_layers = self.model_config.get('configuration_hidden_layers', [512])
        self.num_hidden_layers = len(self.configuration_hidden_layers)

        # --- Partie CNN pour traiter l'observation de forme (4, 84, 84) ---
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=self.observation_space.shape[0], out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )

        # Calcul dynamique de la dimension de sortie du CNN via un dummy input
        with torch.no_grad():
            dummy_input = torch.zeros(1, *self.observation_space.shape)
            cnn_output = self.cnn(dummy_input)
            cnn_output_size = cnn_output.shape[1]

        # --- Partie dense pour transformer les features extraites en Q-valeurs ---
        output_size = self.action_space.n
        dense_layers = []
        # Première couche dense : du vecteur CNN vers la première couche cachée
        dense_layers.append(nn.Linear(cnn_output_size, self.configuration_hidden_layers[0]))
        dense_layers.append(self.activation_function)

        # Couches cachées supplémentaires
        for i in range(self.num_hidden_layers - 1):
            dense_layers.append(nn.Linear(self.configuration_hidden_layers[i], self.configuration_hidden_layers[i + 1]))
            dense_layers.append(self.activation_function)

        # Couche de sortie donnant les Q-valeurs
        dense_layers.append(nn.Linear(self.configuration_hidden_layers[-1], output_size))
        self.critic_layers = nn.Sequential(*dense_layers)

    # def forward(self, batch: dict, **kwargs):
    #     """
    #     Passage avant utilisé pendant l'entraînement.
    #     Extrait d'abord les features via le CNN, puis calcule les Q-valeurs via les couches denses.
    #     """
    #     # On suppose que l'observation est dans batch["obs"]
    #     if "obs" not in batch:
    #         raise ValueError("Le batch doit contenir la clé 'obs' pour les observations")
    #     x = batch["obs"]
    #     features = self.cnn(x)
    #     q_values = self.critic_layers(features)
    #     return q_values

    @override(TargetNetworkAPI)
    def forward_target(self, batch: dict) -> torch.Tensor:
        """
        Passage avant pour le réseau cible.
        Dans ce cas, on réutilise la même architecture.
        """
        if "obs" not in batch:
            raise ValueError("Le batch doit contenir la clé 'obs' pour les observations")
        x = batch["obs"]
        features = self.cnn(x)
        q_values = self.critic_layers(features)
        return q_values

    def make_target_networks(self) -> None:
        """
        Méthode à implémenter pour créer les réseaux cibles.
        """
        pass

    def get_target_network_pairs(self):
        """
        Doit retourner une liste de paires (réseau principal, réseau cible) pour synchronisation.
        """
        pass
