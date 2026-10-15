from pathlib import Path
from torch.nn import LeakyReLU

from cubeball import GameModeRange, GameModeTeamRange
from cubeball.reward_functions import GoalReward

from maxalaar_reinforcement_learning.configuration.experimentation_configuration import ExperimentationConfiguration
from maxalaar_reinforcement_learning.architectures.dense_lstm_ppo import DenseLstmPPO

lstm_curriculum_4 = ExperimentationConfiguration(
    experimentation_name='cubeball/lstm/lstm_curriculum_4',
    environment_name='Cubeball',
)


game_mode_range = GameModeRange(
    level_size=((5, 4, 7.5), (15, 4, 20)),
    cuboid_field_margin=((6, 0, 6), (6, 0, 6)),
    goal_size=((3, 4, 5), (3, 4, 5)),
    ball_number=(1, 1),
    obstacle_number=(0, 0),
    max_duration_seconds=(30, 30),
    max_goal=(1, 1),
    team_list=[
        GameModeTeamRange(players_number=(2, 3)),
        GameModeTeamRange(players_number=(2, 3)),
    ],
)

lstm_curriculum_4.environment_configuration = {
    'show_window': False,
    'action_repeat': 8,
    'speedup': 20,
    'reward_function': GoalReward,
    'game_mode_range': game_mode_range,
}

lstm_curriculum_4.ray_local_mode = False

lstm_curriculum_4.reinforcement_learning_configuration.training_name = 'base'
lstm_curriculum_4.reinforcement_learning_configuration.disable_environment_checking = True
lstm_curriculum_4.reinforcement_learning_configuration.flatten_observations = True
lstm_curriculum_4.reinforcement_learning_configuration.compress_observations = False
lstm_curriculum_4.reinforcement_learning_configuration.normalize_actions = True

lstm_curriculum_4.reinforcement_learning_configuration.number_learners = 1
lstm_curriculum_4.reinforcement_learning_configuration.number_gpus_per_learner = 1
lstm_curriculum_4.reinforcement_learning_configuration.number_cpus_per_learner = 1
lstm_curriculum_4.reinforcement_learning_configuration.number_environment_runners = 12
lstm_curriculum_4.reinforcement_learning_configuration.number_gpus_per_environment_runners = 0
lstm_curriculum_4.reinforcement_learning_configuration.number_cpus_per_environment_runners = 1
lstm_curriculum_4.reinforcement_learning_configuration.evaluation_num_environment_runners = 0

lstm_curriculum_4.reinforcement_learning_configuration.batch_mode = 'complete_episodes'
lstm_curriculum_4.reinforcement_learning_configuration.sample_timeout_s = 60
lstm_curriculum_4.reinforcement_learning_configuration.train_batch_size = 2048 * 4
lstm_curriculum_4.reinforcement_learning_configuration.minibatch_size = 512
lstm_curriculum_4.reinforcement_learning_configuration.clip_policy_parameter = 0.2
lstm_curriculum_4.reinforcement_learning_configuration.entropy_coefficient = 0.005

lstm_curriculum_4.reinforcement_learning_configuration.architecture = DenseLstmPPO
lstm_curriculum_4.reinforcement_learning_configuration.architecture_configuration = {
    'configuration_pre_lstm': [512, 256, 128],
    'lstm_cell_size': 128,
    'configuration_post_lstm': [128, 64],
    'activation_function_class': LeakyReLU,
    'max_seq_len': 128,
}

lstm_curriculum_4.reinforcement_learning_configuration.is_multi_agents = True
lstm_curriculum_4.reinforcement_learning_configuration.policies = {
    'shared_policy',
}
def my_policy_mapping_fn(agent_id, episode, **kwargs):
    return 'shared_policy'

lstm_curriculum_4.reinforcement_learning_configuration.policy_mapping_function = my_policy_mapping_fn
lstm_curriculum_4.reinforcement_learning_configuration.evaluation_interval = 50
lstm_curriculum_4.reinforcement_learning_configuration.checkpoint_frequency = 50
lstm_curriculum_4.reinforcement_learning_configuration.evaluation_duration = 10
lstm_curriculum_4.reinforcement_learning_configuration.checkpoint_score_attribute = 'env_runners/num_episodes_lifetime'

lstm_curriculum_4.reinforcement_learning_configuration.initialization_checkpoint_path = Path("/media/SSD_1To/Programing_Projects/Python/Clustering_Agent_Latent_Space/experiments/cubeball/lstm/lstm_curriculum_3/reinforcement_learning/base/version_0/PPO_Cubeball_fd3ca_00000_0_2026-08-05_18-59-15/checkpoint_000020")

# Display Episodes
lstm_curriculum_4.display_episodes_configuration.checkpoint_score_attribute = 'env_runners/num_episodes_lifetime'
lstm_curriculum_4.display_episodes_configuration.checkpoint_score_order = 'max'
lstm_curriculum_4.display_episodes_configuration.environment_configuration = {
    'game_mode_range': game_mode_range,
}
lstm_curriculum_4.display_episodes_configuration.log_actions = False
