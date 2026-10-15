from torch.nn import LeakyReLU

from cubeball import GameModeRange, GameModeTeamRange
from cubeball.reward_functions import GoalReward

from maxalaar_reinforcement_learning.configuration.experimentation_configuration import ExperimentationConfiguration
from maxalaar_reinforcement_learning.architectures.transformer_ppo import TransformerPPO

transformer_1 = ExperimentationConfiguration(
    experimentation_name='cubeball/transformer/transformer_1',
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

transformer_1.environment_configuration = {
    'show_window': False,
    'action_repeat': 8,
    'speedup': 20,
    'reward_function': GoalReward,
    'game_mode_range': game_mode_range,
    'observation_mode': 'token',
}

transformer_1.ray_local_mode = False

transformer_1.reinforcement_learning_configuration.training_name = 'base'
transformer_1.reinforcement_learning_configuration.disable_environment_checking = True
transformer_1.reinforcement_learning_configuration.flatten_observations = False
transformer_1.reinforcement_learning_configuration.compress_observations = False
transformer_1.reinforcement_learning_configuration.normalize_actions = True

transformer_1.reinforcement_learning_configuration.number_learners = 1
transformer_1.reinforcement_learning_configuration.number_gpus_per_learner = 1
transformer_1.reinforcement_learning_configuration.number_cpus_per_learner = 1
transformer_1.reinforcement_learning_configuration.number_environment_runners = 10
transformer_1.reinforcement_learning_configuration.number_gpus_per_environment_runners = 0
transformer_1.reinforcement_learning_configuration.number_cpus_per_environment_runners = 1
transformer_1.reinforcement_learning_configuration.evaluation_num_environment_runners = 0

transformer_1.reinforcement_learning_configuration.batch_mode = 'complete_episodes'
transformer_1.reinforcement_learning_configuration.sample_timeout_s = 60
transformer_1.reinforcement_learning_configuration.train_batch_size = 2048 * 10
transformer_1.reinforcement_learning_configuration.minibatch_size = 512
transformer_1.reinforcement_learning_configuration.clip_policy_parameter = 0.2
transformer_1.reinforcement_learning_configuration.entropy_coefficient = 0.005

transformer_1.reinforcement_learning_configuration.architecture = TransformerPPO
# Same defaults the old (pre-token) transformer_ppo.py used to ship with, just
# expressed with the new per-key config -- see rllib_repertory/architectures/transformer_ppo.py.
transformer_1.reinforcement_learning_configuration.architecture_configuration = {
    'activation_function_class': LeakyReLU,
    'token_projector_layer_shape': [32, 16],
    'number_heads': 1,
    'dimension_feedforward': 32,
    'number_transformer_layers': 1,
    'action_dense_layer_shapes': [128, 64, 32, 16],
    'critic_dense_layer_shapes': [128, 64, 32, 16],
}

transformer_1.reinforcement_learning_configuration.is_multi_agents = True
transformer_1.reinforcement_learning_configuration.policies = {
    'shared_policy',
}
def my_policy_mapping_fn(agent_id, episode, **kwargs):
    return 'shared_policy'

transformer_1.reinforcement_learning_configuration.policy_mapping_function = my_policy_mapping_fn
transformer_1.reinforcement_learning_configuration.evaluation_interval = 50
transformer_1.reinforcement_learning_configuration.checkpoint_frequency = 50
transformer_1.reinforcement_learning_configuration.evaluation_duration = 10
transformer_1.reinforcement_learning_configuration.checkpoint_score_attribute = 'env_runners/num_episodes_lifetime'

# TransformerPPO's weights are not compatible with the DensePPO checkpoints from
# dense/curriculum_1, so no initialization_checkpoint_path is set here.

# Display Episodes
transformer_1.display_episodes_configuration.checkpoint_score_attribute = 'env_runners/num_episodes_lifetime'
transformer_1.display_episodes_configuration.checkpoint_score_order = 'max'
transformer_1.display_episodes_configuration.environment_configuration = {
    "game_mode_range": game_mode_range,
    "disable_goal_animation": False,
    "disable_ui": False,
    "disable_goal_nets": False,
    "disable_cameras": False,
    "disable_environment": False,
}
transformer_1.display_episodes_configuration.log_actions = False
