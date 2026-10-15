import numbers
import time
from typing import Optional, Dict

import gymnasium as gym
from gymnasium import Space
import numpy as np
from ray.rllib import MultiAgentEnv

RLGym = None
LookupTableAction = None
RepeatAction = None
GoalCondition = None
NoTouchTimeoutCondition = None
TimeoutCondition = None
AnyCondition = None
DefaultObs = None
CombinedReward = None
GoalReward = None
TouchReward = None
RocketSimEngine = None
MutatorSequence = None
FixedTeamSizeMutator = None
KickoffMutator = None
common_values = None
RLViserRenderer = None
rlviser = None
try:
    from rlgym.api import RLGym
    from rlgym.rocket_league.action_parsers import LookupTableAction, RepeatAction
    from rlgym.rocket_league.done_conditions import GoalCondition, NoTouchTimeoutCondition, TimeoutCondition, AnyCondition
    from rlgym.rocket_league.obs_builders import DefaultObs
    from rlgym.rocket_league.reward_functions import CombinedReward, GoalReward, TouchReward
    from rlgym.rocket_league.sim import RocketSimEngine
    from rlgym.rocket_league.state_mutators import MutatorSequence, FixedTeamSizeMutator, KickoffMutator
    from rlgym.rocket_league import common_values
    from rlgym.rocket_league.rlviser import RLViserRenderer

    # from rlgym_ppo.util import RLGymV2GymWrapper
    import rlviser_py as rlviser
    # import RocketSim as rsim
except ImportError:
    print('Rocket League is not installed.')


def make_rocket_league_simulation(environment_configuration: dict = {}):
    spawn_opponents = environment_configuration.get('spawn_opponents', True)
    team_size = environment_configuration.get('team_size', 2)
    blue_team_size = team_size
    orange_team_size = team_size if spawn_opponents else 0
    action_repeat = environment_configuration.get('action_repeat', 8)

    action_parser = RepeatAction(
        LookupTableAction(),
        repeats=action_repeat
    )

    default_observation_builder = DefaultObs(
        zero_padding=None,
        pos_coef=np.asarray(
            [
                1 / common_values.SIDE_WALL_X,
                1 / common_values.BACK_NET_Y,
                1 / common_values.CEILING_Z
            ]
        ),
        ang_coef=1 / np.pi,
        lin_vel_coef=1 / common_values.CAR_MAX_SPEED,
        ang_vel_coef=1 / common_values.CAR_MAX_ANG_VEL,
        boost_coef=1 / 100.0,
    )

    reward_function = environment_configuration.get('reward_function', TouchReward())
    termination_condition = environment_configuration.get('termination_condition', TimeoutCondition(timeout_seconds=300))
    truncation_condition = environment_configuration.get('truncation_condition', None)
    observation_builder = environment_configuration.get('observation_builder', default_observation_builder)

    state_mutator = MutatorSequence(
        FixedTeamSizeMutator(
            blue_size=blue_team_size,
            orange_size=orange_team_size
        ),
        KickoffMutator()
    )
    rlgym_env = RLGym(
        state_mutator=state_mutator,
        obs_builder=observation_builder,
        action_parser=action_parser,
        reward_fn=reward_function,
        termination_cond=termination_condition,
        truncation_cond=truncation_condition,
        transition_engine=RocketSimEngine(),
        renderer=RLViserRenderer(),
    )

    return rlgym_env


def get_gymnasium_space(type: str, size: int) -> Space:
    if type == 'real':
        space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(size,),
            dtype=np.float64
        )
    elif type == 'discrete':
        space = gym.spaces.Discrete(
            n=size,
        )
    else:
        raise ValueError(f"Invalid type '{type}'. Expected 'real' or 'discrete'.")

    return space


def process_actions(actions: Dict):
    for agent_name in actions.keys():
        if isinstance(actions[agent_name], numbers.Integral):
            actions[agent_name] = np.array([actions[agent_name],])

    return actions


class RocketLeague(MultiAgentEnv):
    def __init__(self, environment_configuration: Optional[dict] = None):
        super().__init__()

        self.metadata = {
            'render_modes': ['human'],
        }
        self.render_mode = environment_configuration.get('render_mode', None)
        self.time_sleep = environment_configuration.get('time_sleep', 1/30)

        self.agents = []
        self.possible_agents = []
        self.observation_spaces = {}
        self.action_spaces = {}

        self.simulation = make_rocket_league_simulation(environment_configuration)
        self.simulation.reset()

        for agent_name in self.simulation.agents:
            self.agents.append(agent_name)
            self.possible_agents.append(agent_name)

            observation_type, observation_shape = self.simulation.observation_spaces[agent_name]
            action_type, action_shape = self.simulation.action_spaces[agent_name]

            self.observation_spaces[agent_name] = get_gymnasium_space(observation_type, observation_shape)
            self.action_spaces[agent_name] = get_gymnasium_space(action_type, action_shape)

        self.observation_space = gym.spaces.Dict(self.observation_spaces)
        self.action_space = gym.spaces.Dict(self.action_spaces)

    def reset(self, *, seed=None, options=None):
        observation = self.simulation.reset()
        information = {}
        for agent_name in self.simulation.agents:
            information[agent_name] = {"state": self.simulation.state}

        return observation, information

    def step(self, actions):
        actions = process_actions(actions)
        observation, reward, done, truncated = self.simulation.step(actions)
        done["__all__"] = done[list(done.keys())[0]]
        truncated["__all__"] = truncated[list(truncated.keys())[0]]

        information = {}
        for agent_name in self.simulation.agents:
            information[agent_name] = {"state": self.simulation.state}

        if self.render_mode is not None:
            self.simulation.render()
            time.sleep(self.time_sleep)

        return observation, reward, done, truncated, information

    def render(self):
        self.simulation.render()

    def seed(self, seed=None):
        pass

    def close(self):
        self.simulation.close()

