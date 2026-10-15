from typing import Dict, Any, Optional, Type
import numpy as np
import gymnasium as gym
import torch
from torch import nn, TensorType
from ray.rllib.core.rl_module.apis import ValueFunctionAPI
from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.core.models.catalog import Catalog
from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.models.torch.torch_distributions import TorchDistribution, TorchDiagGaussian
from ray.rllib.utils.annotations import override
from ray.rllib.utils.numpy import MIN_LOG_NN_OUTPUT, MAX_LOG_NN_OUTPUT
from ray.rllib.core.columns import Columns

from maxalaar_reinforcement_learning.neural_network.activation_functions import activation_functions
from maxalaar_reinforcement_learning.neural_network.create_dense_architecture import create_dense_architecture
from maxalaar_reinforcement_learning.training.is_env_runner_process import is_env_runner_process

# `TorchDiagGaussian.from_logits` (used by Catalog for continuous `Box` sub-actions) applies
# no bound on the network's raw log-std output, unlike RLlib's own legacy Gaussian
# distributions which clamp it to [MIN_LOG_NN_OUTPUT, MAX_LOG_NN_OUTPUT]. Without this, the
# entropy bonus keeps pushing log-std up during training until `exp(log_std)` overflows,
# producing `nan`/`inf` gradients. Patch it once here with the same bounds RLlib uses elsewhere.
@classmethod
def _clamped_from_logits(cls, logits: TensorType, **kwargs) -> 'TorchDiagGaussian':
    loc, log_std = logits.chunk(2, dim=-1)
    log_std = torch.clamp(log_std, MIN_LOG_NN_OUTPUT, MAX_LOG_NN_OUTPUT)
    return cls(loc=loc, scale=log_std.exp())


TorchDiagGaussian.from_logits = _clamped_from_logits


class DensePPO(TorchRLModule, ValueFunctionAPI):
    @override(TorchRLModule)
    def get_inference_action_dist_cls(self) -> Type[TorchDistribution]:
        # Ray's default only resolves `Discrete`/`Box` action spaces. Reuse Ray's own
        # Catalog resolution so composite spaces (`Dict`/`Tuple`, e.g. Godot envs'
        # per-action `Discrete` dict) get a matching multi-distribution too.
        return Catalog._get_dist_cls_from_action_space(
            action_space=self.action_space,
            framework='torch',
        )

    @override(TorchRLModule)
    def setup(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.activation_function_class = self.model_config.get('activation_function_class', nn.ReLU)
        self.layer_normalization = self.model_config.get('layer_normalization', False)
        self.dropout = self.model_config.get('dropout', False)
        self.configuration_hidden_layers = self.model_config.get('configuration_hidden_layers', [64, 64])
        self.num_hidden_layers = len(self.configuration_hidden_layers)
        self.use_same_encoder_actor_critic = self.model_config.get('use_same_encoder_actor_critic', False)
        self.configuration_encoder_hidden_layers = self.model_config.get('configuration_encoder_hidden_layers', [64, 64])

        # Récupération du flag pour le mode création de dataset
        self.recorder_mode = self.model_config.get('recorder_mode', False)

        # Dictionnaire pour stocker les activations via hooks
        self.activations = {}

        if not is_env_runner_process():
            print('Observation Space : ' + str(self.observation_space))
            print('Observation Space shape : ' + str(self.observation_space.shape))
            print('Action Space : ' + str(self.action_space))

        self.inpout_size = get_preprocessor(self.observation_space)(self.observation_space).size
        self.output_size = self.get_inference_action_dist_cls().required_input_dim(space=self.action_space)

        if self.use_same_encoder_actor_critic:
            self.encoder_layers = create_dense_architecture(
                self.inpout_size,
                self.configuration_encoder_hidden_layers[:-1],
                self.configuration_encoder_hidden_layers[-1],
                self.activation_function_class,
                dropout=self.dropout,
            )
            self.actor_layers = create_dense_architecture(
                self.configuration_encoder_hidden_layers[-1],
                self.configuration_hidden_layers,
                self.output_size,
                self.activation_function_class,
                dropout=self.dropout,
            )
            self.critic_layers = create_dense_architecture(
                self.configuration_encoder_hidden_layers[-1],
                self.configuration_hidden_layers,
                1,
                self.activation_function_class,
                dropout=self.dropout,
            )
        else:
            self.actor_layers = create_dense_architecture(
                self.inpout_size,
                self.configuration_hidden_layers,
                self.output_size,
                self.activation_function_class,
                dropout=self.dropout,
            )
            self.critic_layers = create_dense_architecture(
                self.inpout_size,
                self.configuration_hidden_layers,
                1,
                self.activation_function_class,
                dropout=self.dropout,
            )
        if not is_env_runner_process():
            print(self)

        # En mode dataset, on enregistre les hooks pour capturer les activations
        if self.recorder_mode:
            self.initialisation_hooks()

        self.to(self.device)

    def initialisation_hooks(self):
        # Enregistrement des hooks
        self._register_hooks()

    def _register_hooks(self):
        def save_activation(module, input, output, name):
            # Stocker la sortie sans gradient
            self.activations[str(name) + '_' + str(module)] = output.detach()

        # Enregistrer un hook pour chaque module Linear
        for name, module in self.named_modules():
            # print(str(name) + ' : ' + str(module))
            if isinstance(module, activation_functions):
                module.register_forward_hook(lambda m, i, o, name=name: save_activation(m, i, o, name))

    @override(TorchRLModule)
    def _forward(self, batch, **kwargs):
        obs = batch[Columns.OBS].to(self.device)

        if self.use_same_encoder_actor_critic:
            encoder_output = self.encoder_layers(obs)
            action_distribution_inputs = self.actor_layers(encoder_output)
        else:
            action_distribution_inputs = self.actor_layers(obs)

        output = {
            Columns.ACTION_DIST_INPUTS: action_distribution_inputs,
        }

        if self.recorder_mode:
            self.activations['input'] = obs
            self.activations['actor_output'] = action_distribution_inputs.to(self.device)

            if self.use_same_encoder_actor_critic:
                critic_output = self.critic_layers(self.encoder_layers(obs)).squeeze(-1)
            else:
                critic_output = self.critic_layers(obs).squeeze(-1)
            self.activations['critic_output'] = critic_output

            output['activations'] = {key: value.detach().cpu().numpy() for key, value in self.activations.items()}
            output['critic_values'] = critic_output.detach().cpu().numpy()

        return output

    def forward_inference_logits(self, obs_dict: dict) -> dict:
        # Raw actor logits, before any action distribution: what the ncnn export
        # carries (the action is decoded by the caller), and the base of `forward_onnx`.
        obs = obs_dict['obs']

        if self.use_same_encoder_actor_critic:
            logits = self.actor_layers(self.encoder_layers(obs))
        else:
            logits = self.actor_layers(obs)

        return {'logits': logits}

    def forward_onnx(self, obs_dict: dict) -> dict:
        logits = self.forward_inference_logits(obs_dict)['logits']

        if not isinstance(self.action_space, gym.spaces.Dict):
            return {'output': logits}

        action = (
            self.get_inference_action_dist_cls()
            .from_logits(logits)
            .to_deterministic()
            .sample()
        )

        return action

    @override(ValueFunctionAPI)
    def compute_values(
            self,
            batch: Dict[str, Any],
            embeddings: Optional[Any] = None,
    ) -> TensorType:
        obs = batch[Columns.OBS].to(self.device)
        if self.use_same_encoder_actor_critic:
            return self.critic_layers(self.encoder_layers(obs)).squeeze(-1)
        else:
            return self.critic_layers(obs).squeeze(-1)
