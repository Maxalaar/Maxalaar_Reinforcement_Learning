import argparse
import os
from pathlib import Path
import warnings

import ray
from ray.rllib.algorithms import Algorithm, AlgorithmConfig

from maxalaar_reinforcement_learning.configuration.experimentation_configuration import ExperimentationConfiguration
from maxalaar_reinforcement_learning.environments.register_environments import register_environment, register_environments
from maxalaar_reinforcement_learning.checkpoints.find_best_checkpoint_path import find_best_reinforcement_learning_checkpoint_path
from maxalaar_reinforcement_learning.checkpoints.get_checkpoint_algorithm_configuration import get_checkpoint_algorithm_configuration
from maxalaar_reinforcement_learning.environments.register_human_rendering_environment_creator import register_human_rendering_environment_creator
from maxalaar_reinforcement_learning.utilities.initialize_ray import initialize_ray
from maxalaar_reinforcement_learning.utilities.save.get_configuration_class import get_configuration_class
from maxalaar_reinforcement_learning.neural_network.neural_network_visualizer_callbacks import create_neural_network_visualizer_callback
from maxalaar_reinforcement_learning.connectors.log_observations import LogObservations
from maxalaar_reinforcement_learning.connectors.log_actions import LogActions


def display_episodes_with_human_rendering(
    experimentation_configuration: ExperimentationConfiguration,
    reinforcement_learning_path: Path,
):
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    register_environments()

    display_episodes_configuration = experimentation_configuration.display_episodes_configuration

    initialize_ray(
        experimentation_configuration,
        local_mode=display_episodes_configuration.force_ray_local_mode or experimentation_configuration.ray_local_mode,
    )

    if display_episodes_configuration.working_directory is not None:
        os.chdir(display_episodes_configuration.working_directory)
        print(os.getcwd())

    if experimentation_configuration.display_episodes_configuration.checkpoint_score_attribute is None:
        metric = experimentation_configuration.reinforcement_learning_configuration.checkpoint_score_attribute
        mode = experimentation_configuration.reinforcement_learning_configuration.checkpoint_score_order
    else:
        metric = experimentation_configuration.display_episodes_configuration.checkpoint_score_attribute
        mode = experimentation_configuration.display_episodes_configuration.checkpoint_score_order

    best_checkpoints_path: Path = find_best_reinforcement_learning_checkpoint_path(
        metric,
        mode,
        reinforcement_learning_path,
    )
    algorithm_configuration = get_checkpoint_algorithm_configuration(best_checkpoints_path)

    register_environment(algorithm_configuration.env)  # a class path name is not among the registered ones
    human_rendering_environment_name = register_human_rendering_environment_creator(
        environment_name=algorithm_configuration.env,
    )

    algorithm_configuration.environment(
        env=human_rendering_environment_name,
    )

    algorithm_configuration.env_config.update(display_episodes_configuration.environment_configuration)

    algorithm_configuration.learners(
        num_learners=0,
        num_gpus_per_learner=0,
    )

    algorithm_configuration.env_runners(
        num_env_runners=0,
        num_envs_per_env_runner=1,
        num_cpus_per_env_runner=1,
        num_gpus_per_env_runner=display_episodes_configuration.num_gpus_per_env_runner,
    )

    evaluation_env_config = {
        'render_mode': display_episodes_configuration.render_mode,
    }
    evaluation_env_config.update(display_episodes_configuration.environment_configuration)

    algorithm_configuration.evaluation(
        evaluation_num_env_runners=1,
        evaluation_duration=1000,
        evaluation_parallel_to_training=False,
        evaluation_duration_unit='episodes',
        evaluation_sample_timeout_s=60*60,
        evaluation_config=AlgorithmConfig.overrides(
            explore=display_episodes_configuration.explore,
            env_config=evaluation_env_config,
        ),
    )

    if display_episodes_configuration.log_observations or display_episodes_configuration.log_actions:
        max_logs = display_episodes_configuration.log_max_steps

        if display_episodes_configuration.log_observations:
            existing_env_to_module = algorithm_configuration._env_to_module_connector

            def make_env_to_module(env=None, spaces=None, **kwargs):
                connectors = []
                if existing_env_to_module is not None:
                    result = existing_env_to_module(env, spaces, **kwargs)
                    connectors.extend(result if isinstance(result, list) else [result])
                connectors.append(LogObservations(max_logs=max_logs))
                return connectors

            algorithm_configuration.env_runners(env_to_module_connector=make_env_to_module)

        if display_episodes_configuration.log_actions:
            existing_module_to_env = algorithm_configuration._module_to_env_connector

            def make_module_to_env(env=None, spaces=None, **kwargs):
                connectors = []
                if existing_module_to_env is not None:
                    result = existing_module_to_env(env, spaces, **kwargs)
                    connectors.extend(result if isinstance(result, list) else [result])
                connectors.append(LogActions(max_logs=max_logs))
                return connectors

            algorithm_configuration.env_runners(module_to_env_connector=make_module_to_env)

    if display_episodes_configuration.use_neural_network_visualizer:
        algorithm_configuration.callbacks(
            create_neural_network_visualizer_callback(
                is_multi_agents = experimentation_configuration.reinforcement_learning_configuration.is_multi_agents,
                max_connections_per_layer = display_episodes_configuration.max_connections_per_layer,
                policies_name = display_episodes_configuration.policies_name,
                policies_name_to_batch_indices_to_visualize= display_episodes_configuration.policies_name_to_batch_indices_to_visualize,
            )
        )

    algorithm: Algorithm = algorithm_configuration.build()

    try:
        algorithm.restore(str(best_checkpoints_path))
        information = algorithm.evaluate()
    finally:
        # Ensures the eval EnvRunner's `stop()` is called (closes the Godot
        # connection cleanly via a "close" message) instead of the process
        # being killed and leaving the Godot subprocess orphaned, which
        # happens when this script is interrupted (e.g. Ctrl+C) while
        # `evaluate()` is still running.
        algorithm.stop()
        ray.shutdown()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate episode human_rendering from policy.')

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
    experimentation_configuration = get_configuration_class(arguments.experimentation_configuration_file)

    reinforcement_learning_path = Path(arguments.reinforcement_learning_path)
    if not reinforcement_learning_path.is_absolute():
        reinforcement_learning_path = Path.cwd() / reinforcement_learning_path

    display_episodes_with_human_rendering(
        experimentation_configuration=experimentation_configuration,
        reinforcement_learning_path=reinforcement_learning_path,
    )
