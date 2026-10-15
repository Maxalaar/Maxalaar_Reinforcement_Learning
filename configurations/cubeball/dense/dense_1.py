from pathlib import Path
import torch
from ray.rllib.utils.from_config import NotProvided
from torch.nn import LeakyReLU

from cubeball import GameModeRange, GameModeTeamRange
from cubeball.reward_functions import GoalReward

from maxalaar_reinforcement_learning.configuration.experimentation_configuration import ExperimentationConfiguration
from maxalaar_reinforcement_learning.architectures.dense_ppo import DensePPO
from maxalaar_reinforcement_learning.callbacks.memory_usage_callback import MemoryUsageCallback

dense_1 = ExperimentationConfiguration(
    experimentation_name='cubeball/dense_1',
    environment_name='Cubeball',
)

game_mode_range = GameModeRange(
    level_size=((5, 4, 7.5), (5, 4, 7.5)),
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

dense_1.environment_configuration = {
    'game_mode_range': game_mode_range,
    'reward_function': GoalReward,
}

dense_1.ray_local_mode = False

dense_1.reinforcement_learning_configuration.training_name = 'base'
dense_1.reinforcement_learning_configuration.disable_environment_checking = True
dense_1.reinforcement_learning_configuration.flatten_observations = True
dense_1.reinforcement_learning_configuration.compress_observations = False
dense_1.reinforcement_learning_configuration.normalize_actions = True
dense_1.reinforcement_learning_configuration.callback = MemoryUsageCallback

dense_1.reinforcement_learning_configuration.number_learners = 1
dense_1.reinforcement_learning_configuration.number_gpus_per_learner = 1
dense_1.reinforcement_learning_configuration.number_cpus_per_learner = 1
dense_1.reinforcement_learning_configuration.number_environment_runners = 10
dense_1.reinforcement_learning_configuration.number_gpus_per_environment_runners = 0
dense_1.reinforcement_learning_configuration.number_cpus_per_environment_runners = 1
dense_1.reinforcement_learning_configuration.evaluation_num_environment_runners = 0

dense_1.reinforcement_learning_configuration.batch_mode = 'complete_episodes'
dense_1.reinforcement_learning_configuration.sample_timeout_s = 60
dense_1.reinforcement_learning_configuration.train_batch_size = 2048 * 10
dense_1.reinforcement_learning_configuration.minibatch_size = 512
dense_1.reinforcement_learning_configuration.clip_policy_parameter = 0.2
dense_1.reinforcement_learning_configuration.entropy_coefficient = 0.005

dense_1.reinforcement_learning_configuration.initialization_checkpoint_path = Path("/media/SSD_1To/Programing_Projects/Python/Clustering_Agent_Latent_Space/experiments/cubeball/dense_1/reinforcement_learning/base/version_2/PPO_Cubeball_26a38_00000_0_2026-08-20_21-12-56/checkpoint_000009")

dense_1.reinforcement_learning_configuration.architecture = DensePPO
dense_1.reinforcement_learning_configuration.architecture_configuration = {
    'configuration_hidden_layers': [1024, 512, 256, 128],
    'activation_function_class': LeakyReLU,
}

dense_1.reinforcement_learning_configuration.is_multi_agents = True
dense_1.reinforcement_learning_configuration.policies = {
    'shared_policy',
}
def my_policy_mapping_fn(agent_id, episode, **kwargs):
    return 'shared_policy'

dense_1.reinforcement_learning_configuration.policy_mapping_function = my_policy_mapping_fn
dense_1.reinforcement_learning_configuration.evaluation_interval = 50
dense_1.reinforcement_learning_configuration.checkpoint_frequency = 50
dense_1.reinforcement_learning_configuration.evaluation_duration = 10
dense_1.reinforcement_learning_configuration.checkpoint_score_attribute = 'env_runners/num_episodes_lifetime'

# Display Episodes
dense_1.display_episodes_configuration.checkpoint_score_attribute = 'env_runners/num_episodes_lifetime'
dense_1.display_episodes_configuration.checkpoint_score_order = 'max'
dense_1.display_episodes_configuration.environment_configuration = {
    'game_mode_range': game_mode_range,
    "disable_goal_animation": False,
    "disable_ui": False,
    "disable_goal_nets": False,
    "disable_cameras": False,
    "disable_environment": False,
}
dense_1.display_episodes_configuration.log_actions = False
