import math
from typing import List, Dict, Any

import numpy as np
from rlgym.api import RewardFunction, AgentID
from rlgym.rocket_league.api import GameState
from rlgym.rocket_league import common_values


class GoalAvoidanceReward(RewardFunction[AgentID, GameState, float]):
    def __init__(self):
        self.max_distance = math.sqrt(common_values.SIDE_WALL_X ** 2 + common_values.BACK_NET_Y ** 2 + common_values.CEILING_Z ** 2)

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        pass

    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        return {agent: self._get_reward(agent, state) for agent in agents}

    def _get_reward(self, agent: AgentID, state: GameState) -> float:
        agent_team = state.cars[agent].team_num

        if agent_team == common_values.BLUE_TEAM:
            goal_position = common_values.BLUE_GOAL_CENTER
        else:
            goal_position = common_values.ORANGE_GOAL_CENTER

        ball_position = state.ball.position
        distance = np.linalg.norm(goal_position - ball_position)
        return float(distance / self.max_distance)