from pathlib import Path
import torch
from ray.rllib.utils.from_config import NotProvided
from torch.nn import LeakyReLU

from cubeball import GameModeRange, GameModeTeamRange
from cubeball.reward_functions import GoalReward

from maxalaar_reinforcement_learning.configuration.experimentation_configuration import ExperimentationConfiguration
from maxalaar_reinforcement_learning.architectures.dense_ppo import DensePPO
from maxalaar_reinforcement_learning.callbacks.memory_usage_callback import MemoryUsageCallback

dense_2 = ExperimentationConfiguration(
    experimentation_name='cubeball/dense_2',
    environment_name='Cubeball',
)

game_mode_range = GameModeRange(
    level_size=((5, 4, 7.5), (10, 4, 15)),
    cuboid_field_margin=((6, 0, 6), (6, 0, 6)),
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

dense_2.environment_configuration = {
    'game_mode_range': game_mode_range,
    'reward_function': GoalReward,
}

dense_2.ray_local_mode = False

dense_2.reinforcement_learning_configuration.training_name = 'base'
dense_2.reinforcement_learning_configuration.disable_environment_checking = True
dense_2.reinforcement_learning_configuration.flatten_observations = True
dense_2.reinforcement_learning_configuration.compress_observations = False
dense_2.reinforcement_learning_configuration.normalize_actions = True
dense_2.reinforcement_learning_configuration.callback = MemoryUsageCallback

dense_2.reinforcement_learning_configuration.number_learners = 1
dense_2.reinforcement_learning_configuration.number_gpus_per_learner = 1
dense_2.reinforcement_learning_configuration.number_cpus_per_learner = 1
dense_2.reinforcement_learning_configuration.number_environment_runners = 10
dense_2.reinforcement_learning_configuration.number_gpus_per_environment_runners = 0
dense_2.reinforcement_learning_configuration.number_cpus_per_environment_runners = 1
dense_2.reinforcement_learning_configuration.evaluation_num_environment_runners = 0

dense_2.reinforcement_learning_configuration.batch_mode = 'complete_episodes'
dense_2.reinforcement_learning_configuration.sample_timeout_s = 60
dense_2.reinforcement_learning_configuration.train_batch_size = 2048 * 10
dense_2.reinforcement_learning_configuration.minibatch_size = 512
dense_2.reinforcement_learning_configuration.clip_policy_parameter = 0.2
dense_2.reinforcement_learning_configuration.entropy_coefficient = 0.005

dense_2.reinforcement_learning_configuration.initialization_checkpoint_path = Path("/media/SSD_1To/Programing_Projects/Python/Clustering_Agent_Latent_Space/experiments/cubeball/dense_1/reinforcement_learning/base/version_3/PPO_Cubeball_7a83b_00000_0_2026-08-21_20-09-40/checkpoint_000000")

dense_2.reinforcement_learning_configuration.architecture = DensePPO
dense_2.reinforcement_learning_configuration.architecture_configuration = {
    'configuration_hidden_layers': [1024, 512, 256, 128],
    'activation_function_class': LeakyReLU,
}

dense_2.reinforcement_learning_configuration.is_multi_agents = True
dense_2.reinforcement_learning_configuration.policies = {
    'shared_policy',
}
def my_policy_mapping_fn(agent_id, episode, **kwargs):
    return 'shared_policy'

dense_2.reinforcement_learning_configuration.policy_mapping_function = my_policy_mapping_fn
dense_2.reinforcement_learning_configuration.evaluation_interval = 50
dense_2.reinforcement_learning_configuration.checkpoint_frequency = 50
dense_2.reinforcement_learning_configuration.evaluation_duration = 10
dense_2.reinforcement_learning_configuration.checkpoint_score_attribute = 'env_runners/num_episodes_lifetime'

# Display Episodes
dense_2.display_episodes_configuration.checkpoint_score_attribute = 'env_runners/num_episodes_lifetime'
dense_2.display_episodes_configuration.checkpoint_score_order = 'max'
dense_2.display_episodes_configuration.environment_configuration = {
    'game_mode_range': game_mode_range,
    "disable_goal_animation": True,
    "disable_ui": True,
    "disable_goal_nets": False,
    "disable_cameras": False,
    "disable_environment": False,
}
dense_2.display_episodes_configuration.log_actions = False
