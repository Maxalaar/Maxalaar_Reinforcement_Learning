from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from typing import List
import torch
from torch import nn, Tensor
from ray.rllib.core.rl_module.apis import ValueFunctionAPI
from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.core.columns import Columns


class LayerNorm(nn.Module):
    def __init__(self, dim, ch_first=True):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.ch_first = ch_first

    def forward(self, x):
        if self.ch_first:  # b, c, h, w
            return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        else:  # b, h, w, c
            return self.norm(x)


class Block(nn.Module):
    def __init__(self, dim, layer_scale_init_value=1e-6):
        super().__init__()

        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, ch_first=False)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)

        if self.gamma is not None:
            x = self.gamma * x

        x = x.permute(0, 3, 1, 2)
        x = input + x
        return x


@dataclass
class AdvancedCNNSamyConfig:
    in_channels: int = 12
    dims: List[int] = field(default_factory=lambda: [24, 12, 6])
    depths: List[int] = field(default_factory=lambda: [3, 9, 3])
    layer_scale_init_value: float = 1e-6


class AdvancedCNNSamy(TorchRLModule, ValueFunctionAPI):
    def setup(self):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.channel_last = self.model_config.get('channel_last', False)
        output_size = self.get_inference_action_dist_cls().required_input_dim(space=self.action_space)

        config = AdvancedCNNSamyConfig()

        self.num_dims = len(config.dims)
        self.downsample_layers = nn.ModuleList()

        stem = nn.Sequential(
            nn.Conv2d(config.in_channels, config.dims[0], kernel_size=4, stride=4),
            LayerNorm(config.dims[0], ch_first=True)
        )
        self.downsample_layers.append(stem)

        for i in range(len(config.dims) - 1):
            downsample_layer = nn.Sequential(
                LayerNorm(config.dims[i], ch_first=True),
                nn.Conv2d(config.dims[i], config.dims[i + 1], kernel_size=2, stride=2)
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()

        for i in range(len(config.dims)):
            stage = nn.Sequential(
                *[Block(dim=config.dims[i],
                        layer_scale_init_value=config.layer_scale_init_value) for _ in range(config.depths[i])]
            )
            self.stages.append(stage)

        self.norm = LayerNorm(dim=config.dims[-1], ch_first=False)

        self.action_proj = nn.Sequential(
            nn.Linear(config.dims[-1], 256),
            nn.GELU(),
            nn.Linear(256, output_size)
        )

        self.value_proj = nn.Sequential(
            nn.Linear(config.dims[-1], 256),
            nn.GELU(),
            nn.Linear(256, 1)
        )

        self.apply(self._init_weights)

        self.to(self.device)
        print(self)

    def _forward(self, batch, **kwargs):
        obs = batch[Columns.OBS].to(self.device)

        if self.channel_last:
            obs = obs.permute(0, 3, 1, 2)

        x = obs
        for i in range(self.num_dims):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        x = x.mean([-2, -1])

        output = {
            Columns.ACTION_DIST_INPUTS: self.action_proj(x),
        }
        return output

    def _forward_train(self, batch, **kwargs):
        obs = batch[Columns.OBS].to(self.device)

        if self.channel_last:
            obs = obs.permute(0, 3, 1, 2)

        x = obs
        for i in range(self.num_dims):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        x = x.mean([-2, -1])

        output = {
            Columns.ACTION_DIST_INPUTS: self.action_proj(x),
            Columns.EMBEDDINGS: x
        }
        return output

    def compute_values(self, batch: Dict[str, Any], embeddings: Optional[Any] = None) -> Tensor:
        if embeddings is None:
            obs = batch[Columns.OBS].to(self.device)

            if self.channel_last:
                obs = obs.permute(0, 3, 1, 2)

            x = obs
            for i in range(self.num_dims):
                x = self.downsample_layers[i](x)
                x = self.stages[i](x)
            embeddings = x.mean([-2, -1])

        return self.value_proj(embeddings).squeeze(-1)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)