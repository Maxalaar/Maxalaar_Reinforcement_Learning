import argparse
from pathlib import Path
import warnings

from ray.rllib.algorithms import Algorithm, AlgorithmConfig

from maxalaar_reinforcement_learning.configuration.experimentation_configuration import ExperimentationConfiguration
from maxalaar_reinforcement_learning.environments.register_environments import register_environments
from maxalaar_reinforcement_learning.checkpoints.find_best_checkpoint_path import find_best_reinforcement_learning_checkpoint_path
from maxalaar_reinforcement_learning.checkpoints.get_checkpoint_algorithm_configuration import get_checkpoint_algorithm_configuration
from maxalaar_reinforcement_learning.environments.register_video_environment_creator import register_video_environment_creator
from maxalaar_reinforcement_learning.utilities.initialize_ray import initialize_ray
from maxalaar_reinforcement_learning.utilities.save.get_configuration_class import get_configuration_class
from maxalaar_reinforcement_learning.utilities.save.get_reinforcement_learning_name import get_reinforcement_learning_name


def video_episode_generation(experimentation_configuration: ExperimentationConfiguration, reinforcement_learning_path: Path):
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    initialize_ray(experimentation_configuration)
    register_environments()

    metric = experimentation_configuration.reinforcement_learning_configuration.checkpoint_score_attribute
    mode = experimentation_configuration.reinforcement_learning_configuration.checkpoint_score_order
    best_checkpoints_path: Path = find_best_reinforcement_learning_checkpoint_path(metric, mode, reinforcement_learning_path)
    algorithm_configuration = get_checkpoint_algorithm_configuration(best_checkpoints_path)
    algorithm_configuration.env_config.update(experimentation_configuration.video_episodes_generation_configuration.environment_configuration)

    reinforcement_learning_name = get_reinforcement_learning_name(experimentation_configuration, reinforcement_learning_path)
    video_environment_name = register_video_environment_creator(
        environment_name=algorithm_configuration.env,
        video_episodes_storage_path=experimentation_configuration.video_episodes_storage_path / reinforcement_learning_name,
    )

    algorithm_configuration.environment(
        env=video_environment_name
    )
    algorithm_configuration.env_config.update({'render_mode': 'rgb_array'})
    algorithm_configuration.learners(
        num_learners=0,
        num_gpus_per_learner=0,
    )

    algorithm_configuration.env_runners(
        num_env_runners=0,
        num_envs_per_env_runner=experimentation_configuration.video_episodes_generation_configuration.number_environment_per_environment_runners,
        num_cpus_per_env_runner=experimentation_configuration.video_episodes_generation_configuration.number_cpus_per_environment_runners,
        num_gpus_per_env_runner=experimentation_configuration.video_episodes_generation_configuration.number_gpus_per_environment_runners,
    )

    algorithm_configuration.evaluation(
        evaluation_num_env_runners=experimentation_configuration.video_episodes_generation_configuration.number_environment_runners,
        evaluation_duration=experimentation_configuration.video_episodes_generation_configuration.minimal_number_videos,
        evaluation_parallel_to_training=False,
        evaluation_duration_unit='episodes',
        evaluation_sample_timeout_s=60*10,
        evaluation_config=AlgorithmConfig.overrides(explore=False),
    )

    algorithm: Algorithm = algorithm_configuration.build()

    algorithm.restore(str(best_checkpoints_path))
    information = algorithm.evaluate()

    # print('The average reward on the evaluation is :' + str(information['env_runners']['episode_return_mean']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate episode video from policy.')
    parser.add_argument(
        '--experimentation_configuration_file',
        type=str,
        help="The path of the experimentation configuration file (e.g., './configurations/experimentation/cartpole.py')"
    )

    parser.add_argument(
        '--reinforcement_learning_path',
        type=str,
        help="The path of repository with the reinforcement learning checkpoint (e.g., './experiments/cartpole/reinforcement_learning/base')"
    )

    arguments = parser.parse_args()
    configuration_class = get_configuration_class(arguments.experimentation_configuration_file)

    reinforcement_learning_path = Path(arguments.reinforcement_learning_path)
    if not reinforcement_learning_path.is_absolute():
        reinforcement_learning_path = Path.cwd() / reinforcement_learning_path

    video_episode_generation(configuration_class, reinforcement_learning_path)
