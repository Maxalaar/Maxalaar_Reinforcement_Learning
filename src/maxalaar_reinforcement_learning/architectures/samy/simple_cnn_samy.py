from typing import Dict, Any, Optional
import torch
from torch import nn, Tensor
from ray.rllib.core.rl_module.apis import ValueFunctionAPI
from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.core.columns import Columns

class SimpleCNNSamy(TorchRLModule, ValueFunctionAPI):
    def setup(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.channel_last = self.model_config.get('channel_last', False)
        output_size = self.get_inference_action_dist_cls().required_input_dim(space=self.action_space)

        self.embedder = nn.Sequential(
            nn.Conv2d(in_channels=24, out_channels=32, kernel_size=8, stride=4),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=4, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )

        self.projector = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(16 * 6 * 6, 256),
            nn.RMSNorm(256),
            nn.ReLU(),
        )
        self.projector_action = nn.Linear(256, output_size)
        self.projector_value = nn.Linear(256, 1)

        self.to(self.device)
        print(self)

    def _forward(self, batch, **kwargs):
        obs = batch[Columns.OBS].to(self.device)
        if self.channel_last:
            obs = obs.permute(0, 3, 1, 2)
        cnn_features = self.embedder(obs)
        embeddings = self.projector(cnn_features)

        output = {
            Columns.ACTION_DIST_INPUTS: self.projector_action(embeddings),
        }
        return output

    def _forward_train(self, batch, **kwargs):
        obs = batch[Columns.OBS].to(self.device)
        if self.channel_last:
            obs = obs.permute(0, 3, 1, 2)
        cnn_features = self.embedder(obs)
        embeddings = self.projector(cnn_features)
        action_distribution_inputs = self.projector_action(embeddings)

        output = {
            Columns.ACTION_DIST_INPUTS: action_distribution_inputs,
            Columns.EMBEDDINGS: embeddings
        }
        return output

    def compute_values(self, batch: Dict[str, Any], embeddings: Optional[Any] = None) -> Tensor:

        if embeddings is None:
            obs = batch[Columns.OBS].to(self.device)
            if self.channel_last:
                obs = obs.permute(0, 3, 1, 2)
            cnn_features = self.embedder(obs)
            embeddings = self.projector(cnn_features)

        return self.projector_value(embeddings).squeeze(-1)