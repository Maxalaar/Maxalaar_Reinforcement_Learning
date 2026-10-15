from typing import Dict, Any, Optional, Type, List

import torch
from torch import nn, TensorType
from ray.rllib.core.rl_module.apis import ValueFunctionAPI
from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.core.models.catalog import Catalog
from ray.rllib.models.torch.torch_distributions import TorchDistribution
from ray.rllib.utils.annotations import override
from ray.rllib.core.columns import Columns

from maxalaar_reinforcement_learning.neural_network.create_dense_architecture import create_dense_architecture
from maxalaar_reinforcement_learning.training.is_env_runner_process import is_env_runner_process


class TransformerPPO(TorchRLModule, ValueFunctionAPI):
    # Expects a Dict observation space where each non-"*_mask" key is either a
    # "solo" token (1D Box, never padded, e.g. "self"/"game_state") or a padded
    # group of tokens sharing one projector (2D Box of shape (max_count, features)
    # with a companion "<key>_mask" MultiBinary(max_count), e.g. "cuboid"/"ball").
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

        self.number_heads = self.model_config.get('number_heads', 1)
        self.dimension_feedforward = self.model_config.get('dimension_feedforward', 32)
        self.number_transformer_layers = self.model_config.get('number_transformer_layers', 1)

        # Every token key shares this same dense architecture (only the first
        # layer's input width differs, fixed by that key's raw feature count --
        # see _build_token_projectors). Its last value is the token size fed into
        # the self-attention, e.g. [64, 32] means two hidden layers then a token
        # of size 32.
        self.token_projector_layer_shape: List[int] = self.model_config.get(
            'token_projector_layer_shape', [64, 32]
        )
        self.dimension_token: int = self.token_projector_layer_shape[-1]

        action_dense_layer_shapes = self.model_config.get('action_dense_layer_shapes', [128, 64, 32, 16])
        critic_dense_layer_shapes = self.model_config.get('critic_dense_layer_shapes', [128, 64, 32, 16])

        self.token_keys, self.group_keys, self.token_input_dimensions = self._classify_observation_space()

        self._build_token_stack('actor')
        self._build_token_stack('critic')

        self.action_dense_layers = create_dense_architecture(
            input_dimension=self.dimension_token,
            configuration_hidden_layers=action_dense_layer_shapes,
            output_dimension=self.get_inference_action_dist_cls().required_input_dim(space=self.action_space),
            activation_function_class=self.activation_function_class,
        )
        self.critic_dense_layers = create_dense_architecture(
            input_dimension=self.dimension_token,
            configuration_hidden_layers=critic_dense_layer_shapes,
            output_dimension=1,
            activation_function_class=self.activation_function_class,
        )

        self.to(self.device)
        if not is_env_runner_process():
            print(self)

    # Splits self.observation_space into: solo token keys (1D Box), group token
    # keys (2D Box, paired with a "<key>_mask"), and each key's raw feature width
    # -- "*_mask" keys themselves are skipped here, they're read directly by key
    # in _encode.
    def _classify_observation_space(self):
        token_keys: List[str] = []
        group_keys: List[str] = []
        token_input_dimensions: Dict[str, int] = {}

        for key, space in self.observation_space.spaces.items():
            if key.endswith('_mask'):
                continue

            if len(space.shape) == 1:
                token_input_dimensions[key] = space.shape[0]
            elif len(space.shape) == 2:
                group_keys.append(key)
                token_input_dimensions[key] = space.shape[1]
            else:
                raise ValueError(f"Unsupported observation shape for key '{key}': {space.shape}")

            token_keys.append(key)

        return token_keys, group_keys, token_input_dimensions

    # Every key gets the same token_projector_layer_shape architecture -- only the
    # first layer's input width differs, fixed by that key's raw feature count.
    def _build_token_projectors(self) -> nn.ModuleDict:
        projectors = {
            key: create_dense_architecture(
                input_dimension=self.token_input_dimensions[key],
                configuration_hidden_layers=self.token_projector_layer_shape[:-1],
                output_dimension=self.dimension_token,
                activation_function_class=self.activation_function_class,
            )
            for key in self.token_keys
        }

        return nn.ModuleDict(projectors)

    # Builds one full {projectors, learned context token, transformer encoder}
    # stack under self.<prefix>_* -- called once for "actor" and once for
    # "critic", always fully separate (no shared-encoder option).
    def _build_token_stack(self, prefix: str) -> None:
        setattr(self, f'{prefix}_token_projectors', self._build_token_projectors())

        assert self.dimension_token % self.number_heads == 0, (
            f"dimension_token ({self.dimension_token}, the last entry of token_projector_layer_shape) "
            f"must be divisible by number_heads ({self.number_heads})"
        )

        transformer_layer = nn.TransformerEncoderLayer(
            d_model=self.dimension_token,
            nhead=self.number_heads,
            dim_feedforward=self.dimension_feedforward,
            dropout=0.0,
            batch_first=True,
        )
        setattr(
            self,
            f'{prefix}_transformer_encoder',
            nn.TransformerEncoder(transformer_layer, num_layers=self.number_transformer_layers),
        )
        setattr(self, f'{prefix}_context_token', nn.Parameter(torch.randn(self.dimension_token), requires_grad=True))

    # Projects every observation key into a token, prepends the learned context
    # token, runs the self-attention (masking out padded group slots), and
    # returns the context token's output -- the embedding the dense heads decide
    # from. Entities within a group are an unordered set (e.g. "other cuboid #2"
    # is not meaningfully before/after "#0"), so unlike the old per-scalar
    # version there is no positional encoding here: self-attention is already
    # permutation-equivariant, which is exactly the property we want, and adding
    # a position would just teach the model to rely on an arbitrary spawn order.
    def _encode(self, observation: Dict[str, TensorType], prefix: str) -> TensorType:
        projectors: nn.ModuleDict = getattr(self, f'{prefix}_token_projectors')
        context_token: nn.Parameter = getattr(self, f'{prefix}_context_token')
        transformer_encoder: nn.TransformerEncoder = getattr(self, f'{prefix}_transformer_encoder')

        batch_size = observation[self.token_keys[0]].shape[0]

        tokens = [context_token.expand(batch_size, 1, self.dimension_token)]
        padding_masks = [torch.zeros(batch_size, 1, dtype=torch.bool, device=self.device)]

        for key in self.token_keys:
            value = observation[key].to(self.device).float()

            if key in self.group_keys:
                batch, count, dimension = value.shape
                projected = projectors[key](value.reshape(batch * count, dimension))
                projected = projected.reshape(batch, count, self.dimension_token)
                padding_masks.append(observation[f'{key}_mask'].to(self.device) == 0)
            else:
                projected = projectors[key](value).unsqueeze(1)
                padding_masks.append(torch.zeros(batch_size, 1, dtype=torch.bool, device=self.device))

            tokens.append(projected)

        tokens = torch.cat(tokens, dim=1)
        src_key_padding_mask = torch.cat(padding_masks, dim=1)

        transformer_output = transformer_encoder(tokens, src_key_padding_mask=src_key_padding_mask)

        return transformer_output[:, 0]

    @override(TorchRLModule)
    def _forward(self, batch, **kwargs):
        embedding = self._encode(batch[Columns.OBS], 'actor')

        return {
            Columns.ACTION_DIST_INPUTS: self.action_dense_layers(embedding),
        }

    @override(ValueFunctionAPI)
    def compute_values(
            self,
            batch: Dict[str, Any],
            embeddings: Optional[Any] = None,
    ) -> TensorType:
        embedding = self._encode(batch[Columns.OBS], 'critic')

        return self.critic_dense_layers(embedding).squeeze(-1)
