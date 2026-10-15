from typing import Any, Dict, List, Optional

import numpy as np

from ray.rllib.connectors.connector_v2 import ConnectorV2
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import EpisodeType


class LogObservations(ConnectorV2):
    """Env-to-module connector that prints each agent's observation for debugging."""

    def __init__(
        self,
        input_observation_space=None,
        input_action_space=None,
        *,
        max_logs: Optional[int] = None,
        **kwargs,
    ):
        self._log_count = 0
        self._max_logs = max_logs
        super().__init__(input_observation_space, input_action_space, **kwargs)

    @override(ConnectorV2)
    def __call__(
        self,
        *,
        rl_module: RLModule,
        batch: Dict[str, Any],
        episodes: List[EpisodeType],
        explore: Optional[bool] = None,
        shared_data: Optional[dict] = None,
        **kwargs,
    ) -> Any:
        if self._max_logs is not None and self._log_count >= self._max_logs:
            return batch

        for sa_episode in self.single_agent_episode_iterator(
            episodes, agents_that_stepped_only=True
        ):
            obs = sa_episode.get_observations(-1)
            agent_id = getattr(sa_episode, 'agent_id', 'default')
            obs_array = np.asarray(obs)
            print(
                f'[OBS] agent={agent_id} | shape={obs_array.shape} | {obs_array}'
            )
            self._log_count += 1

        return batch
