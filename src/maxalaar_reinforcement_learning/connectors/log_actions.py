from typing import Any, Dict, List, Optional

from ray.rllib.connectors.connector_v2 import ConnectorV2
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import EpisodeType


class LogActions(ConnectorV2):
    """Module-to-env connector that prints each agent's action for debugging."""

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
        batch: Optional[Dict[str, Any]],
        episodes: List[EpisodeType],
        explore: Optional[bool] = None,
        shared_data: Optional[dict] = None,
        **kwargs,
    ) -> Any:
        if self._max_logs is not None and self._log_count >= self._max_logs:
            return batch

        if Columns.ACTIONS not in batch:
            return batch

        def _log_and_return(action, env_id, agent_id, module_id):
            print(
                f'[ACTION] env={env_id} | agent={agent_id} | module={module_id} | action={action}'
            )
            self._log_count += 1
            return action

        self.foreach_batch_item_change_in_place(
            batch=batch,
            column=Columns.ACTIONS,
            func=_log_and_return,
        )

        return batch
