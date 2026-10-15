from typing import TYPE_CHECKING
from typing import Union, Optional, Dict, List
import tkinter as tk

import gymnasium as gym
from ray.rllib import BaseEnv, Policy
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.core.rl_module import RLModule, MultiRLModule
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.utils.metrics.metrics_logger import MetricsLogger
from ray.rllib.utils.typing import EpisodeType, PolicyID

from maxalaar_reinforcement_learning.neural_network.neural_network_visualizer import NeuralNetworkVisualizer

if TYPE_CHECKING:
    from ray.rllib.env.env_runner import EnvRunner


def create_neural_network_visualizer_callback(
        is_multi_agents: bool,
        max_connections_per_layer: int,
        policies_name: List[str],
        policies_name_to_batch_indices_to_visualize: Optional[Dict[str, int]],
):

    class NeuralNetworkVisualizerCallbacks(DefaultCallbacks):
        def __init__(self):
            root = tk.Tk()
            root.withdraw()

            self.is_multi_agents = is_multi_agents
            self.max_connections_per_layer = max_connections_per_layer
            self.neural_network_visualizers: Optional[List[NeuralNetworkVisualizer]] = None
            self.policies_name = policies_name
            self.policies_name_to_batch_indices_to_visualize = policies_name_to_batch_indices_to_visualize

        def on_episode_step(
                self,
                *,
                episode: Union[EpisodeType, EpisodeV2],
                env_runner: Optional["EnvRunner"] = None,
                metrics_logger: Optional[MetricsLogger] = None,
                env: Optional[gym.Env] = None,
                env_index: int,
                rl_module: Optional[RLModule] = None,
                worker: Optional["EnvRunner"] = None,
                base_env: Optional[BaseEnv] = None,
                policies: Optional[Dict[PolicyID, Policy]] = None,
                **kwargs,
        ) -> None:
            if self.neural_network_visualizers is None:
                self.neural_network_visualizers = []
                if self.is_multi_agents:
                    multi_rl_module: MultiRLModule = rl_module
                    for policy_name in policies_name:
                        rl_module: RLModule = multi_rl_module[policy_name]
                        if policies_name_to_batch_indices_to_visualize is not None and policy_name in policies_name_to_batch_indices_to_visualize.keys():
                            batch_index_to_visualize = policies_name_to_batch_indices_to_visualize[policy_name]
                        else:
                            batch_index_to_visualize = 0
                        self.neural_network_visualizers.append(
                            NeuralNetworkVisualizer(
                                rl_module.actor_layers,
                                max_connections_per_layer=self.max_connections_per_layer,
                                batch_index_to_visualize=batch_index_to_visualize,
                                window_title=policy_name,
                            )
                        )
                else:
                    rl_module: RLModule = rl_module


                    if policies_name_to_batch_indices_to_visualize is not None:
                        batch_index_to_visualize = 0
                    else:
                        batch_index_to_visualize = 0

                    self.neural_network_visualizers.append(
                        NeuralNetworkVisualizer(
                            rl_module.actor_layers,
                            max_connections_per_layer=self.max_connections_per_layer,
                            batch_index_to_visualize=batch_index_to_visualize,
                        )
                    )

            for neural_network_visualizer in self.neural_network_visualizers:
                neural_network_visualizer.update()

    return NeuralNetworkVisualizerCallbacks