import argparse
import os
from pathlib import Path

from ray import tune

from maxalaar_reinforcement_learning.configuration.experimentation_configuration import ExperimentationConfiguration
from maxalaar_reinforcement_learning.environments.register_environments import register_environments
from maxalaar_reinforcement_learning.utilities.initialize_ray import initialize_ray
from maxalaar_reinforcement_learning.training.build_algorithm_configuration import build_algorithm_configuration
from maxalaar_reinforcement_learning.utilities.save.get_configuration_class import get_configuration_class


def continue_reinforcement_learning_training(
        experimentation_configuration: ExperimentationConfiguration,
        reinforcement_learning_path: Path,
):
    initialize_ray(experimentation_configuration)
    register_environments()
    reinforcement_learning_configuration = experimentation_configuration.reinforcement_learning_configuration

    tuner_save_path = os.path.join(str(reinforcement_learning_path), 'tuner.pkl')
    if not os.path.exists(tuner_save_path):
        raise FileNotFoundError(
            f"Tuner checkpoint not found at: {tuner_save_path}. "
            f"Make sure the path points to a version directory (e.g., .../version_0/) "
            f"that contains a tuner.pkl file."
        )

    # Rebuild the algorithm configuration from the (possibly updated) experimentation
    # configuration file, so config changes made since the run started (e.g. a new
    # callback) are picked up on resume instead of silently reusing the pickled config.
    algorithm, algorithm_configuration = build_algorithm_configuration(experimentation_configuration)

    tuner = tune.Tuner.restore(
        os.path.dirname(tuner_save_path),
        trainable=algorithm,
        param_space=algorithm_configuration,
        resume_unfinished=True,
        resume_errored=True,
        restart_errored=True,
    )

    tuner.fit()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Continue an interrupted reinforcement learning training.')
    parser.add_argument(
        '--experimentation_configuration_file',
        type=str,
        help="The path of the experimentation configuration file (e.g., './configurations/experimentation/cartpole.py')"
    )
    parser.add_argument(
        '--reinforcement_learning_path',
        type=str,
        required=True,
        help="Path to the version directory of the training to resume (e.g., './experiments/cartpole/reinforcement_learning/base/version_0/')"
    )

    arguments = parser.parse_args()
    configuration_class = get_configuration_class(arguments.experimentation_configuration_file)

    reinforcement_learning_path = Path(arguments.reinforcement_learning_path)
    if not reinforcement_learning_path.is_absolute():
        reinforcement_learning_path = Path.cwd() / reinforcement_learning_path

    continue_reinforcement_learning_training(configuration_class, reinforcement_learning_path)
