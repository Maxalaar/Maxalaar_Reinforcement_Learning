from typing import Dict, Any, Optional
import torch
from torch import nn, Tensor
from ray.rllib.core.rl_module.apis import ValueFunctionAPI
from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.core.columns import Columns

from maxalaar_reinforcement_learning.neural_network.create_cnn_architecture import create_cnn_architecture
from maxalaar_reinforcement_learning.neural_network.create_dense_architecture import create_dense_architecture
from maxalaar_reinforcement_learning.training.is_env_runner_process import is_env_runner_process

# 64*64
# 'configuration_cnn': [(16, 4, 2), (32, 4, 2), (64, 4, 2), (128, 4, 2)],

# 128*128
# 'configuration_cnn': [
#     (16, 4, 2),  # 63x63
#     (32, 4, 2),  # 30x30
#     (64, 4, 2),  # 14x14
#     (128, 4, 2), # 6x6
#     (256, 4, 2)  # 2x2 → Flatten → 256*2*2 = 1024
# ],


class CnnPPO(TorchRLModule, ValueFunctionAPI):
    def setup(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.activation_function_class = self.model_config.get("activation_function_class", nn.ReLU)
        self.configuration_layer_normalizations = self.model_config.get("configuration_layer_normalizations", None)
        self.dropout = self.model_config.get("dropout", False)
        self.configuration_hidden_layers = self.model_config.get("configuration_hidden_layers", [64, 64])
        self.configuration_cnn = self.model_config.get(
            "configuration_cnn",
            [(32, 8, 4), (64, 4, 2), (64, 3, 1)]
        )
        self.use_layer_normalization_cnn = self.model_config.get("use_layer_normalization_cnn", False)
        self.channel_last = self.model_config.get('channel_last', False)
        self.use_share_cnn = self.model_config.get('use_share_cnn', True)

        print('Observation space : ' + str(self.observation_space))
        print('Action space : ' + str(self.action_space))

        # Determine input channels for CNN
        if self.channel_last:
            in_channels = self.observation_space.shape[-1]
        else:
            in_channels = self.observation_space.shape[0] if len(self.observation_space.shape) == 3 else 1

        if self.use_share_cnn:
            self.shared_cnn = create_cnn_architecture(
                in_channels=in_channels,
                configuration_cnn=self.configuration_cnn,
                activation_function_class=self.activation_function_class,
                use_normalization=self.use_layer_normalization_cnn,
            )
        else:
            self.actor_cnn = create_cnn_architecture(
                in_channels=in_channels,
                configuration_cnn=self.configuration_cnn,
                activation_function_class=self.activation_function_class,
                use_normalization=self.use_layer_normalization_cnn,
            )
            self.critic_cnn = create_cnn_architecture(
                in_channels=in_channels,
                configuration_cnn=self.configuration_cnn,
                activation_function_class=self.activation_function_class,
                use_normalization=self.use_layer_normalization_cnn,
            )

        # Dynamically compute CNN output size
        dummy_input = torch.zeros(32, *self.observation_space.shape)
        if self.channel_last:
            dummy_input = dummy_input.permute(0, 3, 1, 2)  # Convert to (batch, channels, x, y)
        if self.use_share_cnn:
            cnn_output_size = self.shared_cnn(dummy_input).shape[1]
        else:
            cnn_output_size = self.actor_cnn(dummy_input).shape[1]
        if not is_env_runner_process():
            print('Observation Space shape : ' + str(self.observation_space.shape))
            print(f'CNN output size: {cnn_output_size}')

        self.output_size = self.get_inference_action_dist_cls().required_input_dim(space=self.action_space)

        # Create dense layers for actor and critic
        self.actor_layers = create_dense_architecture(
            cnn_output_size,
            self.configuration_hidden_layers,
            self.output_size,
            self.activation_function_class,
            configuration_layer_normalizations=self.configuration_layer_normalizations,
            dropout=self.dropout,
        )
        self.critic_layers = create_dense_architecture(
            cnn_output_size,
            self.configuration_hidden_layers,
            1,
            self.activation_function_class,
            configuration_layer_normalizations=self.configuration_layer_normalizations,
            dropout=self.dropout,
        )

        self.to(self.device)
        if not is_env_runner_process():
            print(self)

    def _get_obs_from_batch(self, batch):
        obs = batch[Columns.OBS].to(self.device)

        if obs.dtype != torch.float32:
            obs = obs.to(dtype=torch.float32)

        if self.channel_last:
            obs = obs.permute(0, 3, 1, 2)

        return obs

    def _forward(self, batch, **kwargs):
        obs = self._get_obs_from_batch(batch)

        if self.use_share_cnn:
            cnn_features = self.shared_cnn(obs)
        else:
            cnn_features = self.actor_cnn(obs)

        action_distribution_inputs = self.actor_layers(cnn_features)

        output = {
            Columns.ACTION_DIST_INPUTS: action_distribution_inputs,
        }

        return output

    def _forward_train(self, batch, **kwargs):
        obs = self._get_obs_from_batch(batch)

        if self.use_share_cnn:
            cnn_features = self.shared_cnn(obs)
        else:
            cnn_features = self.actor_cnn(obs)

        embeddings = cnn_features
        action_distribution_inputs = self.actor_layers(cnn_features)

        output = {
            Columns.ACTION_DIST_INPUTS: action_distribution_inputs,
            Columns.EMBEDDINGS: embeddings,
        }

        return output

    def compute_values(self, batch: Dict[str, Any], embeddings: Optional[Any] = None) -> Tensor:
        if embeddings is None:
            obs = self._get_obs_from_batch(batch)

            if self.use_share_cnn:
                cnn_features = self.shared_cnn(obs)
            else:
                cnn_features = self.critic_cnn(obs)
            embeddings = cnn_features

        return self.critic_layers(embeddings).squeeze(-1)
