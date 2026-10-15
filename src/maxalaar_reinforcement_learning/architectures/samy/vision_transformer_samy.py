from dataclasses import dataclass
from typing import Dict, Any, Optional
import torch
from torch import nn, Tensor
from ray.rllib.core.rl_module.apis import ValueFunctionAPI
from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.core.columns import Columns
from einops.layers.torch import Rearrange
from einops import rearrange, repeat

from maxalaar_reinforcement_learning.neural_network.create_dense_architecture import create_dense_architecture


class Patcher(nn.Module):
    def __init__(self, channels, img_height, img_width, patch_size, d_model):
        super().__init__()
        assert img_height % patch_size == 0 and img_width % patch_size == 0, f"img_height ({img_height}) and img_width ({img_width}) should be divisible by patch_size ({patch_size})"
        patch_dim = patch_size ** 2 * channels

        self.patcher = nn.Sequential(
            Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=patch_size, p2=patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, d_model),
            nn.LayerNorm(d_model)
        )

    def forward(self, x):
        return self.patcher(x)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dp=0.1):
        super().__init__()
        self.ff = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dp),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dp)
        )

    def forward(self, x):
        return self.ff(x)


class MHSA(nn.Module):
    def __init__(self, d_model, n_heads, dp=0.1):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"

        self.head_dim = d_model // n_heads
        self.n_heads = n_heads
        self.scale = self.head_dim ** -0.5

        self.norm = nn.LayerNorm(d_model)
        self.dp = nn.Dropout(dp)

        self.to_qkv = nn.Linear(d_model, 3 * d_model)

        self.to_out = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Dropout(dp)
        )

    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.n_heads), qkv)
        sdp = torch.matmul(q, k.transpose(2, 3)) * self.scale
        attn = torch.softmax(sdp, dim=-1)
        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, d_model, n_layers, n_heads, d_ff, dp=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.layers = nn.ModuleList([])
        for _ in range(n_layers):
            self.layers.append(nn.ModuleList([
                MHSA(d_model=d_model, n_heads=n_heads, dp=dp),
                FeedForward(d_model=d_model, d_ff=d_ff, dp=dp)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)

@dataclass
class VitConfig:
    img_height: int = 64
    img_width: int = 64
    channels: int = 12
    patch_size: int = 16
    d_model: int = 128
    n_layers: int = 1
    d_ff: int = 128
    n_heads: int = 4


class VisionTransformerSamy(TorchRLModule, ValueFunctionAPI):
    def setup(self):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.channel_last = self.model_config.get('channel_last', False)
        output_size = self.get_inference_action_dist_cls().required_input_dim(space=self.action_space)

        self.patcher = Patcher(
            channels=VitConfig().channels,
            img_height=VitConfig().img_height,
            img_width=VitConfig().img_width,
            patch_size=VitConfig().patch_size,
            d_model=VitConfig().d_model
        )
        n_patch = int(VitConfig.img_height * VitConfig.img_width / VitConfig.patch_size ** 2)

        self.pos_embedding = nn.Parameter(torch.randn(1, n_patch + 2, VitConfig.d_model))
        self.action_token = nn.Parameter(torch.randn(1, 1, VitConfig.d_model))
        self.value_token = nn.Parameter(torch.randn(1, 1, VitConfig.d_model))

        self.dp = nn.Dropout(0.1)

        self.transformer = Transformer(
            d_model=VitConfig.d_model,
            n_layers=VitConfig.n_layers,
            n_heads=VitConfig.n_heads,
            d_ff=VitConfig.d_ff,
            dp=0.1
        )

        # self.projector_action = nn.Linear(VitConfig.d_model, output_size)
        # self.projector_value = nn.Linear(VitConfig.d_model, 1)

        self.projector_action = create_dense_architecture(
            VitConfig.d_model,
            [256, 256],
            output_size,
            nn.LeakyReLU,
            layer_normalization=False,
            dropout=True,
        )

        self.projector_value = create_dense_architecture(
            VitConfig.d_model,
            [256, 256],
            1,
            nn.LeakyReLU,
            layer_normalization=False,
            dropout=True,
        )

        self.to(self.device)
        print(self)

    def _forward(self, batch, **kwargs):
        obs = batch[Columns.OBS].to(self.device)

        if self.channel_last:
            obs = obs.permute(0, 3, 1, 2)

        x = self.patcher(obs)
        b, n, _ = x.size()

        action_token = repeat(self.action_token, "1 1 d -> b 1 d", b=b)
        value_token = repeat(self.value_token, "1 1 d -> b 1 d", b=b)

        x = torch.cat((action_token, value_token, x), dim=1)
        x += self.pos_embedding[:, :(n + 2), :]
        x = self.dp(x)
        x = self.transformer(x)

        action_embed = x[:, 0, :]

        output = {
            Columns.ACTION_DIST_INPUTS: self.projector_action(action_embed),
        }
        return output

    def _forward_train(self, batch, **kwargs):
        obs = batch[Columns.OBS].to(self.device)

        if self.channel_last:
            obs = obs.permute(0, 3, 1, 2)

        x = self.patcher(obs)
        b, n, _ = x.size()

        action_token = repeat(self.action_token, "1 1 d -> b 1 d", b=b)
        value_token = repeat(self.value_token, "1 1 d -> b 1 d", b=b)

        x = torch.cat((action_token, value_token, x), dim=1)
        x += self.pos_embedding[:, :(n + 2), :]
        x = self.dp(x)
        x = self.transformer(x)

        action_embed = x[:, 0, :]
        value_embed = x[:, 1, :]

        output = {
            Columns.ACTION_DIST_INPUTS: self.projector_action(action_embed),
            Columns.EMBEDDINGS: value_embed
        }
        return output

    def compute_values(self, batch: Dict[str, Any], embeddings: Optional[Any] = None) -> Tensor:
        if embeddings is None:
            obs = batch[Columns.OBS].to(self.device)

            if self.channel_last:
                obs = obs.permute(0, 3, 1, 2)

            x = self.patcher(obs)
            b, n, _ = x.size()

            action_token = repeat(self.action_token, "1 1 d -> b 1 d", b=b)
            value_token = repeat(self.value_token, "1 1 d -> b 1 d", b=b)

            x = torch.cat((action_token, value_token, x), dim=1)
            x += self.pos_embedding[:, :(n + 2), :]
            x = self.dp(x)
            x = self.transformer(x)

            embeddings = x[:, 1, :]
        return self.projector_value(embeddings).squeeze(-1)