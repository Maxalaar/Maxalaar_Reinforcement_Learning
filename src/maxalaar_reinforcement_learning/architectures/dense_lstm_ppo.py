from typing import Dict, Any, Optional, Type
import numpy as np
import gymnasium as gym
import tree
import torch
from torch import nn, Tensor
from ray.rllib.core.rl_module.apis import ValueFunctionAPI
from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.core.models.catalog import Catalog
from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.models.torch.torch_distributions import TorchDistribution
from ray.rllib.utils.annotations import override
from ray.rllib.core.columns import Columns

from maxalaar_reinforcement_learning.neural_network.create_dense_architecture import create_dense_architecture
from maxalaar_reinforcement_learning.training.is_env_runner_process import is_env_runner_process


class DenseLstmPPO(TorchRLModule, ValueFunctionAPI):
    @override(TorchRLModule)
    def get_inference_action_dist_cls(self) -> Type[TorchDistribution]:
        # Ray's default only resolves `Discrete`/`Box` action spaces. Reuse Ray's own
        # Catalog resolution so composite spaces (`Dict`/`Tuple`, e.g. Godot envs'
        # per-action `Discrete` dict) get a matching multi-distribution too.
        composite_distribution_class = Catalog._get_dist_cls_from_action_space(
            action_space=self.action_space,
            framework='torch',
        )

        # Ray's composite `logp()` (as of ray==2.56.0) squeezes away the last
        # dimension of any `Discrete` sub-action whose shape ends in 1 -- meant for
        # 0-d `Box` sub-actions, but it also fires for the legitimate T=1 time-rank
        # that stateful (recurrent) modules carry, which then breaks broadcasting
        # against the child distribution's batch shape. Non-stateful modules (e.g.
        # `DensePPO`) never carry that time-rank, so they never hit this bug. Bypass
        # it here by summing the per-key log-probs ourselves, without the squeeze.
        class FixedCompositeDistribution(composite_distribution_class):
            def logp(self, value):
                if isinstance(value, torch.Tensor):
                    return super().logp(value)
                flat_values = tree.flatten(value)
                flat_logps = [
                    child_distribution.logp(flat_value)
                    for flat_value, child_distribution in zip(flat_values, self._flat_child_distributions)
                ]
                # Some Ray distributions (e.g. `TorchDeterministic`, used for `Box`
                # sub-actions once `to_deterministic()` is applied, i.e. explore=False
                # inference) don't reduce the action's own trailing dimension(s) out
                # of `logp`, unlike their stochastic counterparts. Sum each child's
                # logp down to the common (batch[, time]) shape before combining them.
                reference_number_dimensions = min(logp.dim() for logp in flat_logps)
                flat_logps = [
                    logp.sum(dim=tuple(range(reference_number_dimensions, logp.dim())))
                    if logp.dim() > reference_number_dimensions
                    else logp
                    for logp in flat_logps
                ]
                return sum(flat_logps)

            def to_deterministic(self):
                # Ray's `TorchMultiDistribution.to_deterministic()` hardcodes the
                # returned instance's class to the base `TorchMultiDistribution`,
                # which drops this fix (used by explore=False inference, e.g.
                # `display_episodes.py`). Re-apply the same class-swap trick Ray
                # itself uses in `from_logits` to keep the fixed `logp` in effect.
                deterministic_distribution = super().to_deterministic()
                deterministic_distribution.__class__ = type(self)
                return deterministic_distribution

        return FixedCompositeDistribution

    @override(TorchRLModule)
    def setup(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.activation_function_class = self.model_config.get('activation_function_class', nn.ReLU)

        self.configuration_pre_lstm = self.model_config.get('configuration_pre_lstm', [64, 64])
        self.lstm_cell_size = self.model_config.get('lstm_cell_size', 64)
        self.configuration_post_lstm = self.model_config.get('configuration_post_lstm', [64, 64])
        # Separate actor/critic encoders+LSTMs by default: more stable (no
        # competition between the policy and value gradients over a shared
        # recurrent state) at the cost of double the parameters/compute.
        self.use_same_encoder_actor_critic = self.model_config.get('use_same_encoder_actor_critic', False)

        if not is_env_runner_process():
            print('Observation Space : ' + str(self.observation_space))
            print('Observation Space shape : ' + str(self.observation_space.shape))
            print('Action Space : ' + str(self.action_space))

        self.input_size = get_preprocessor(self.observation_space)(self.observation_space).size
        self.output_size = self.get_inference_action_dist_cls().required_input_dim(space=self.action_space)

        if self.use_same_encoder_actor_critic:
            self.pre_lstm_dense = create_dense_architecture(
                input_dimension=self.input_size,
                configuration_hidden_layers=self.configuration_pre_lstm[:-1],
                output_dimension=self.configuration_pre_lstm[-1],
                activation_function_class=self.activation_function_class,
                dropout=False,
            )
            self.lstm = nn.LSTM(
                input_size=self.configuration_pre_lstm[-1],
                hidden_size=self.lstm_cell_size,
                batch_first=True,
            )
        else:
            self.actor_pre_lstm_dense = create_dense_architecture(
                input_dimension=self.input_size,
                configuration_hidden_layers=self.configuration_pre_lstm[:-1],
                output_dimension=self.configuration_pre_lstm[-1],
                activation_function_class=self.activation_function_class,
                dropout=False,
            )
            self.actor_lstm = nn.LSTM(
                input_size=self.configuration_pre_lstm[-1],
                hidden_size=self.lstm_cell_size,
                batch_first=True,
            )
            self.critic_pre_lstm_dense = create_dense_architecture(
                input_dimension=self.input_size,
                configuration_hidden_layers=self.configuration_pre_lstm[:-1],
                output_dimension=self.configuration_pre_lstm[-1],
                activation_function_class=self.activation_function_class,
                dropout=False,
            )
            self.critic_lstm = nn.LSTM(
                input_size=self.configuration_pre_lstm[-1],
                hidden_size=self.lstm_cell_size,
                batch_first=True,
            )

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

    @override(TorchRLModule)
    def get_initial_state(self) -> Dict[str, Any]:
        zero_state = {
            'h': np.zeros(self.lstm_cell_size, dtype=np.float32),
            'c': np.zeros(self.lstm_cell_size, dtype=np.float32),
        }
        if self.use_same_encoder_actor_critic:
            return zero_state
        return {
            'actor': zero_state,
            'critic': dict(zero_state),
        }

    def _compute_post_lstm_embedding(self, batch, pre_lstm_dense, lstm, state_in):
        obs = batch[Columns.OBS].to(self.device)

        B, T = obs.shape[:2]
        obs = obs.reshape(B * T, *obs.shape[2:])

        pre_lstm_embedding = pre_lstm_dense(obs)
        pre_lstm_embedding = pre_lstm_embedding.view(B, T, -1)

        state_dict = {
            'h': torch.as_tensor(state_in['h'], device=self.device).float(),
            'c': torch.as_tensor(state_in['c'], device=self.device).float(),
        }

        post_lstm_embedding, (h_out, c_out) = lstm(
            pre_lstm_embedding,
            (state_dict['h'].unsqueeze(0), state_dict['c'].unsqueeze(0)),
        )

        state_out = {
            'h': h_out.squeeze(0).detach(),
            'c': c_out.squeeze(0).detach(),
        }

        return post_lstm_embedding, state_out

    def get_post_lstm_embedding(self, batch):
        state_in = batch.get(Columns.STATE_IN, self.get_initial_state())

        if self.use_same_encoder_actor_critic:
            shared_embedding, state_out = self._compute_post_lstm_embedding(
                batch, self.pre_lstm_dense, self.lstm, state_in,
            )
            return shared_embedding, shared_embedding, state_out

        actor_embedding, actor_state_out = self._compute_post_lstm_embedding(
            batch, self.actor_pre_lstm_dense, self.actor_lstm, state_in['actor'],
        )
        critic_embedding, critic_state_out = self._compute_post_lstm_embedding(
            batch, self.critic_pre_lstm_dense, self.critic_lstm, state_in['critic'],
        )
        state_out = {'actor': actor_state_out, 'critic': critic_state_out}

        return actor_embedding, critic_embedding, state_out

    @override(TorchRLModule)
    def _forward(self, batch, **kwargs):
        actor_embedding, critic_embedding, state_out = self.get_post_lstm_embedding(batch)
        action_distribution_inputs = self.actor_layers(actor_embedding)

        output = {
            Columns.ACTION_DIST_INPUTS: action_distribution_inputs,
            Columns.STATE_OUT: state_out,
        }

        return output

    def _forward_train(self, batch, **kwargs):
        actor_embedding, critic_embedding, state_out = self.get_post_lstm_embedding(batch)
        action_distribution_inputs = self.actor_layers(actor_embedding)

        output = {
            Columns.ACTION_DIST_INPUTS: action_distribution_inputs,
            Columns.STATE_OUT: state_out,
            Columns.EMBEDDINGS: critic_embedding,
        }

        return output

    def forward_inference_logits(self, obs_dict: dict) -> dict:
        # Stateful single-step export: the caller (rllib_repertory/export/ncnn_export.py)
        # wires the actor LSTM's hidden state in (state_in_h/state_in_c) and out
        # (state_out_h/state_out_c), so the exported graph carries real memory
        # across calls -- the caller is responsible for feeding back state_out as
        # the next call's state_in, and for resetting it to zero on episode
        # boundaries. Only the actor path is needed here (the export is for
        # action inference, not values).
        obs = obs_dict['obs']

        pre_lstm_dense = self.pre_lstm_dense if self.use_same_encoder_actor_critic else self.actor_pre_lstm_dense
        lstm = self.lstm if self.use_same_encoder_actor_critic else self.actor_lstm

        pre_lstm_embedding = pre_lstm_dense(obs).unsqueeze(1)

        h = obs_dict['state_in_h'].unsqueeze(0)
        c = obs_dict['state_in_c'].unsqueeze(0)

        post_lstm_embedding, (h_out, c_out) = lstm(pre_lstm_embedding, (h, c))
        logits = self.actor_layers(post_lstm_embedding.squeeze(1))

        return {'logits': logits, 'state_out_h': h_out.squeeze(0), 'state_out_c': c_out.squeeze(0)}

    def forward_onnx(self, obs_dict: dict) -> dict:
        inference_outputs = self.forward_inference_logits(obs_dict)
        logits = inference_outputs['logits']
        state_out = {'state_out_h': inference_outputs['state_out_h'], 'state_out_c': inference_outputs['state_out_c']}

        if not isinstance(self.action_space, gym.spaces.Dict):
            return {'output': logits, **state_out}

        action = (
            self.get_inference_action_dist_cls()
            .from_logits(logits)
            .to_deterministic()
            .sample()
        )

        return {**dict(action), **state_out}

    @override(ValueFunctionAPI)
    def compute_values(
            self,
            batch: Dict[str, Any],
            embeddings: Optional[Any] = None,
    ) -> Tensor:
        if embeddings is None:
            _, embeddings, state_out = self.get_post_lstm_embedding(batch)

        return self.critic_layers(embeddings).squeeze(-1)
