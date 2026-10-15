from typing import Dict, Any, Optional, List, Tuple
from ray.rllib.utils.typing import NetworkType
from ray.rllib.core.models.base import ACTOR, ENCODER_OUT
from ray.rllib.core.rl_module.apis import TARGET_NETWORK_ACTION_DIST_INPUTS

import torch
import numpy as np
from ray.rllib.core.learner.utils import make_target_network
from torch import nn, Tensor
from ray.rllib.core.rl_module.apis import ValueFunctionAPI, TargetNetworkAPI
from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.core.columns import Columns

from maxalaar_reinforcement_learning.neural_network.create_cnn_architecture import create_cnn_architecture
from maxalaar_reinforcement_learning.neural_network.create_dense_architecture import create_dense_architecture
from maxalaar_reinforcement_learning.training.is_env_runner_process import is_env_runner_process
from ray.rllib.utils.annotations import override

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

from ray.rllib.algorithms.ppo.torch.default_ppo_torch_rl_module import DefaultPPOTorchRLModule
from ray.rllib.algorithms.appo.torch.default_appo_torch_rl_module import DefaultAPPOTorchRLModule





class CnnLstmAPPO(TorchRLModule, ValueFunctionAPI, TargetNetworkAPI):
    def setup(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.activation_function_class = self.model_config.get('activation_function_class', nn.ReLU)
        self.channel_last = self.model_config.get('channel_last', False)

        self.configuration_cnn = self.model_config.get('configuration_cnn', [(32, 8, 4), (64, 4, 2), (64, 3, 1)])
        self.use_layer_normalization_cnn = self.model_config.get('use_layer_normalization_cnn', True)
        self.configuration_post_cnn = self.model_config.get('configuration_post_cnn', [64])
        self.lstm_cell_size = self.model_config.get('lstm_cell_size', 64)
        self.configuration_post_lstm = self.model_config.get('configuration_post_lstm', [64, 64])
        self.output_size = self.get_inference_action_dist_cls().required_input_dim(space=self.action_space)

        print('Observation space : ' + str(self.observation_space))

        # Determine input channels for CNN
        if self.channel_last:
            in_channels = self.observation_space.shape[-1]
        else:
            in_channels = self.observation_space.shape[0] if len(self.observation_space.shape) == 3 else 1

        self.shared_cnn = create_cnn_architecture(
            in_channels=in_channels,
            configuration_cnn=self.configuration_cnn,
            activation_function_class=self.activation_function_class,
            use_normalization=self.use_layer_normalization_cnn,
        )

        # Dynamically compute CNN output size
        dummy_input = torch.zeros(32, *self.observation_space.shape)
        if self.channel_last:
            dummy_input = dummy_input.permute(0, 3, 1, 2)  # Convert to (batch, channels, x, y)
        self.cnn_output_size = self.shared_cnn(dummy_input).shape[1]
        if not is_env_runner_process():
            print('Observation Space shape : ' + str(self.observation_space.shape))
            print(f'CNN output size: {self.cnn_output_size}')

        self.post_cnn_dense = create_dense_architecture(
            input_dimension=self.cnn_output_size,
            configuration_hidden_layers=self.configuration_post_cnn[:-1],
            output_dimension=self.configuration_post_cnn[-1],
            activation_function_class=self.activation_function_class,
            dropout=False,
        ).to(self.device)

        self.lstm = nn.LSTM(
            input_size=self.configuration_post_cnn[-1],
            hidden_size=self.lstm_cell_size,
            batch_first=True,
        ).to(self.device)

        # Create dense layers for actor and critic
        self.actor_layers = create_dense_architecture(
            input_dimension=self.lstm_cell_size,
            configuration_hidden_layers=self.configuration_post_lstm,
            output_dimension=self.output_size,
            activation_function_class=self.activation_function_class,
            dropout=False,
        )
        self.critic_layers = create_dense_architecture(
            input_dimension=self.lstm_cell_size,
            configuration_hidden_layers=self.configuration_post_lstm,
            output_dimension=1,
            activation_function_class=self.activation_function_class,
            dropout=False,
        )

        self.to(self.device)
        if not is_env_runner_process():
            print(self)

    @override(TargetNetworkAPI)
    def make_target_networks(self):
        self.old_shared_cnn = make_target_network(self.shared_cnn)
        self.old_post_cnn_dense = make_target_network(self.post_cnn_dense)
        self.old_lstm = make_target_network(self.lstm)
        self.old_actor_layers = make_target_network(self.actor_layers)
        self.old_critic_layers = make_target_network(self.critic_layers)

    @override(TargetNetworkAPI)
    def get_target_network_pairs(self) -> List[Tuple[NetworkType, NetworkType]]:
        return [
            (self.shared_cnn, self.old_shared_cnn),
            (self.post_cnn_dense, self.old_post_cnn_dense),
            (self.lstm, self.old_lstm),
            (self.actor_layers, self.old_actor_layers),
            (self.critic_layers, self.old_critic_layers),
        ]

    @override(TorchRLModule)
    def get_initial_state(self) -> Dict[str, np.ndarray]:
        return {
            'h': np.zeros(self.lstm_cell_size, dtype=np.float32),
            'c': np.zeros(self.lstm_cell_size, dtype=np.float32),
        }

    def get_post_cnn_embedding(self, batch, cnn, post_cnn):
        obs = batch[Columns.OBS].to(self.device)

        if self.channel_last:
            obs = obs.permute(0, 1, 4, 2, 3)  # [B, T, C, H, W]

        B, T = obs.shape[:2]
        obs = obs.reshape(B * T, *obs.shape[2:])

        post_cnn_embedding = cnn(obs)
        post_cnn_embedding = post_cnn_embedding.view(B, T, -1)

        post_cnn_embedding = post_cnn(post_cnn_embedding)

        return post_cnn_embedding

    def get_post_lstm_embedding(self, batch, cnn, post_cnn, lstm):
        post_cnn_embedding = self.get_post_cnn_embedding(batch, cnn, post_cnn)

        state_in = batch.get(Columns.STATE_IN, self.get_initial_state())
        state_dict = {
            'h': torch.as_tensor(state_in['h'], device=self.device).float(),
            'c': torch.as_tensor(state_in['c'], device=self.device).float()
        }

        post_lstm_embedding, (h_out, c_out) = lstm(
            post_cnn_embedding,
            (state_dict['h'].unsqueeze(0), state_dict['c'].unsqueeze(0)),
        )

        state_out = {
            'h': h_out.squeeze(0),
            'c': c_out.squeeze(0),
        }

        return post_lstm_embedding, state_out

    def _forward(self, batch, **kwargs):
        post_lstm_embedding, state_out = self.get_post_lstm_embedding(
            batch,
            self.shared_cnn,
            self.post_cnn_dense,
            self.lstm,
        )
        action_distribution_inputs = self.actor_layers(post_lstm_embedding)

        output = {
            Columns.ACTION_DIST_INPUTS: action_distribution_inputs,
            Columns.STATE_OUT: state_out,
        }

        return output

    def _forward_train(self, batch, **kwargs):
        post_lstm_embedding, state_out = self.get_post_lstm_embedding(
            batch,
            self.shared_cnn,
            self.post_cnn_dense,
            self.lstm,
        )
        embeddings = post_lstm_embedding
        action_distribution_inputs = self.actor_layers(post_lstm_embedding)

        output = {
            Columns.ACTION_DIST_INPUTS: action_distribution_inputs,
            Columns.STATE_OUT: state_out,
            Columns.EMBEDDINGS: embeddings,
        }

        return output

    @override(TargetNetworkAPI)
    def forward_target(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        old_post_lstm_embedding, old_state_out = self.get_post_lstm_embedding(
            batch,
            self.old_shared_cnn,
            self.old_post_cnn_dense,
            self.old_lstm,
        )
        old_embeddings = old_post_lstm_embedding
        old_action_distribution_inputs = self.old_actor_layers(old_post_lstm_embedding)

        output = {
            TARGET_NETWORK_ACTION_DIST_INPUTS: old_action_distribution_inputs,
            Columns.STATE_OUT: old_state_out,
            Columns.EMBEDDINGS: old_embeddings,
        }

        return output

    def compute_values(self, batch: Dict[str, Any], embeddings: Optional[Any] = None) -> Tensor:
        if embeddings is None:
            post_lstm_embedding, state_out = self.get_post_lstm_embedding(
                batch,
                self.shared_cnn,
                self.post_cnn_dense,
                self.lstm,
            )
            embeddings = post_lstm_embedding

        return self.critic_layers(embeddings).squeeze(-1)