from pathlib import Path
import torch
from ray.rllib.utils.from_config import NotProvided
from torch.nn import LeakyReLU

from cubeball import GameModeRange, GameModeTeamRange
from cubeball.reward_functions import GoalReward

from maxalaar_reinforcement_learning.configuration.experimentation_configuration import ExperimentationConfiguration
from maxalaar_reinforcement_learning.architectures.dense_ppo import DensePPO

simple_1v1 = ExperimentationConfiguration(
    experimentation_name='cubeball/simple_1v1',
    environment_name='Cubeball',
)

# Domain randomization: a fresh GameMode is sampled from this every episode
# (see Cubeball.reset / GameModeRange.sample). A fixed value is just a
# degenerate range, e.g. max_goal=(1, 1).
game_mode_range = GameModeRange(
    level_size=((5, 4, 7.5), (5, 4, 7.5)),
    cuboid_field_margin=((0, 0, 0), (0, 0, 0)),
    goal_size=((3, 4, 5), (3, 4, 5)),
    ball_number=(1, 1),
    obstacle_number=(0, 0),
    max_duration_seconds=(30, 30),
    max_goal=(1, 1),
    team_list=[
        GameModeTeamRange(players_number=(1, 1)),
        GameModeTeamRange(players_number=(1, 1)),
    ],
)

simple_1v1.environment_configuration = {
    'show_window': False,
    'action_repeat': 8,
    'speedup': 20,
    'reward_function': GoalReward,
    'game_mode_range': game_mode_range,
}

simple_1v1.ray_local_mode = False

simple_1v1.reinforcement_learning_configuration.training_name = 'base'
simple_1v1.reinforcement_learning_configuration.disable_environment_checking = True
simple_1v1.reinforcement_learning_configuration.flatten_observations = True
simple_1v1.reinforcement_learning_configuration.compress_observations = False
simple_1v1.reinforcement_learning_configuration.normalize_actions = True

simple_1v1.reinforcement_learning_configuration.number_learners = 1
simple_1v1.reinforcement_learning_configuration.number_gpus_per_learner = 1
simple_1v1.reinforcement_learning_configuration.number_cpus_per_learner = 1
simple_1v1.reinforcement_learning_configuration.number_environment_runners = 10
simple_1v1.reinforcement_learning_configuration.number_gpus_per_environment_runners = 0
simple_1v1.reinforcement_learning_configuration.number_cpus_per_environment_runners = 1
simple_1v1.reinforcement_learning_configuration.evaluation_num_environment_runners = 0

simple_1v1.reinforcement_learning_configuration.batch_mode = 'complete_episodes'
simple_1v1.reinforcement_learning_configuration.sample_timeout_s = 60
simple_1v1.reinforcement_learning_configuration.train_batch_size = 2048 * 10
simple_1v1.reinforcement_learning_configuration.minibatch_size = 512
simple_1v1.reinforcement_learning_configuration.clip_policy_parameter = 0.2
simple_1v1.reinforcement_learning_configuration.entropy_coefficient = 0.005

simple_1v1.reinforcement_learning_configuration.architecture = DensePPO
simple_1v1.reinforcement_learning_configuration.architecture_configuration = {
    'configuration_hidden_layers': [512, 256, 128],
    'activation_function_class': LeakyReLU,
}

simple_1v1.reinforcement_learning_configuration.is_multi_agents = True
simple_1v1.reinforcement_learning_configuration.policies = {
    'shared_policy',
}
def my_policy_mapping_fn(agent_id, episode, **kwargs):
    return 'shared_policy'

simple_1v1.reinforcement_learning_configuration.policy_mapping_function = my_policy_mapping_fn
simple_1v1.reinforcement_learning_configuration.evaluation_interval = 50
simple_1v1.reinforcement_learning_configuration.checkpoint_frequency = 50
simple_1v1.reinforcement_learning_configuration.evaluation_duration = 10
simple_1v1.reinforcement_learning_configuration.checkpoint_score_attribute = 'env_runners/num_episodes_lifetime'

# Display Episodes
simple_1v1.display_episodes_configuration.checkpoint_score_attribute = 'env_runners/num_episodes_lifetime'
simple_1v1.display_episodes_configuration.checkpoint_score_order = 'max'
simple_1v1.display_episodes_configuration.environment_configuration = {
    'game_mode_range': game_mode_range,
}
simple_1v1.display_episodes_configuration.log_actions = False