import math
import numpy as np
from typing import List, Dict, Any, Tuple

from rlgym.api import AgentID
from rlgym.rocket_league.api import GameState
from rlgym.rocket_league.obs_builders import DefaultObs
from rlgym.rocket_league.common_values import TICKS_PER_SECOND


class DefaultObservationWithTime(DefaultObs):
    """
    Observation builder that extends DefaultObs by adding the current episode time.
    """

    def __init__(
        self,
        zero_padding=3,
        pos_coef=1 / 2300,
        ang_coef=1 / math.pi,
        lin_vel_coef=1 / 2300,
        ang_vel_coef=1 / math.pi,
        pad_timer_coef=1 / 10,
        boost_coef=1 / 100,
    ):
        """
        :param time_coef: Episode time normalization coefficient (default 1/300 for 5-minute matches)
        """
        super().__init__(zero_padding, pos_coef, ang_coef, lin_vel_coef, ang_vel_coef, pad_timer_coef, boost_coef)
        self.initial_tick = None

    def get_obs_space(self, agent: AgentID) -> Tuple[str, int]:
        # Add 1 to the original obs space for the time feature
        base_space = super().get_obs_space(agent)
        if base_space[1] != -1:
            return base_space[0], base_space[1] + 1
        else:
            return base_space

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        super().reset(agents, initial_state, shared_info)
        self.initial_tick = initial_state.tick_count

    def _build_obs(self, agent: AgentID, state: GameState, shared_info: Dict[str, Any]) -> np.ndarray:
        # Get the base observation from parent class
        base_obs = super()._build_obs(agent, state, shared_info)

        # Get current episode time from shared_info
        # Assuming the time is stored in shared_info under 'time' or 'episode_time'
        # You might need to adjust this key based on your setup
        time_elapsed = (state.tick_count - self.initial_tick) / TICKS_PER_SECOND

        # Append time to the base observation
        return np.append(base_obs, time_elapsed)