from typing import Dict, Any, List, Tuple

from ray.rllib.core.rl_module.apis import ValueFunctionAPI, TargetNetworkAPI
from ray.rllib.utils.typing import NetworkType
from torch import nn
from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.utils.annotations import override
from ray.rllib.core.columns import Columns


# class DQNDense(DQNRainbowRLModule):
#     @override(TorchRLModule)
#     def setup(self):
#         self.activation_function = self.model_config.get('activation_function', nn.ReLU())
#         self.configuration_hidden_layers = self.model_config.get('configuration_hidden_layers', [64, 64])
#         self.num_hidden_layers = len(self.configuration_hidden_layers)
#
#         inpout_size = get_preprocessor(self.observation_space)(self.observation_space).size
#         output_size = self.action_space.n
#
#         actor_layers = [nn.Linear(inpout_size, self.configuration_hidden_layers[0]), self.activation_function]
#         critic_layers = [nn.Linear(inpout_size, self.configuration_hidden_layers[0]), self.activation_function]
#
#         for i in range(0, self.num_hidden_layers - 1):
#             actor_layers.append(nn.Linear(self.configuration_hidden_layers[i], self.configuration_hidden_layers[i + 1]))
#             actor_layers.append(self.activation_function)
#
#             critic_layers.append(
#                 nn.Linear(self.configuration_hidden_layers[i], self.configuration_hidden_layers[i + 1]))
#             critic_layers.append(self.activation_function)
#
#         actor_layers.append(nn.Linear(self.configuration_hidden_layers[-1], output_size))
#         critic_layers.append(nn.Linear(self.configuration_hidden_layers[-1], 1))
#
#         self.actor_layers = nn.Sequential(*actor_layers)
#         self.critic_layers = nn.Sequential(*critic_layers)
#
#     @override(TorchRLModule)
#     def _forward(self, batch, **kwargs):
#         action_distribution_inputs = self.actor_layers(batch[Columns.OBS])
#         return action_distribution_inputs

class DenseDQN(TorchRLModule, TargetNetworkAPI):
    @override(TorchRLModule)
    def setup(self):
        self.activation_function = self.model_config.get('activation_function', nn.ReLU())
        self.configuration_hidden_layers = self.model_config.get('configuration_hidden_layers', [64, 64])
        self.num_hidden_layers = len(self.configuration_hidden_layers)

        inpout_size = get_preprocessor(self.observation_space)(self.observation_space).size
        output_size = self.action_space.n

        critic_layers = [nn.Linear(inpout_size, self.configuration_hidden_layers[0]), self.activation_function]

        for i in range(0, self.num_hidden_layers - 1):
            critic_layers.append(
                nn.Linear(self.configuration_hidden_layers[i], self.configuration_hidden_layers[i + 1]))
            critic_layers.append(self.activation_function)

        critic_layers.append(nn.Linear(self.configuration_hidden_layers[-1], output_size))

        self.critic_layers = nn.Sequential(*critic_layers)

    # def _forward(self, batch, **kwargs):
    #     action_distribution_inputs = self.actor_layers(batch[Columns.OBS])
    #     return action_distribution_inputs

    # @override(TorchRLModule)
    # def _forward_inference(self, batch, **kwargs):
    #     action_distribution_inputs = self.actor_layers(batch[Columns.OBS])
    #     return action_distribution_inputs

    # def _forward_exploration(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    #     action_distribution_inputs = self.actor_layers(batch[Columns.OBS])
    #     return action_distribution_inputs

    def make_target_networks(self) -> None:
        pass

    def get_target_network_pairs(self) -> List[Tuple[NetworkType, NetworkType]]:
        pass

    @override(TargetNetworkAPI)
    def forward_target(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        # return {'af': self.critic_layers(batch[Columns.OBS])}
        return self.critic_layers(batch[Columns.OBS])
