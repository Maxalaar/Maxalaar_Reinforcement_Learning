import argparse
from pathlib import Path

from maxalaar_reinforcement_learning.configuration.experimentation_configuration import ExperimentationConfiguration
from maxalaar_reinforcement_learning.environments.register_environments import register_environments
from maxalaar_reinforcement_learning.training.run_training import create_training_version, start_training
from maxalaar_reinforcement_learning.utilities.initialize_ray import initialize_ray
from maxalaar_reinforcement_learning.utilities.save.get_configuration_class import get_configuration_class


def reinforcement_learning_training(
        experimentation_configuration: ExperimentationConfiguration,
):
    initialize_ray(experimentation_configuration)
    register_environments()
    start_training(experimentation_configuration, create_training_version(experimentation_configuration))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run reinforcement learning training.')
    parser.add_argument(
        '--experimentation_configuration_file',
        type=str,
        help="The path of the experimentation configuration file (e.g., './configurations/experimentation/cartpole.py')"
    )

    parser.add_argument(
        '--initialization_checkpoint_path',
        type=str,
        default=None,
        help="A checkpoint whose module weights start the training, instead of the configuration's "
             "initialization_checkpoint_path (e.g., './experiments/cartpole/reinforcement_learning/base/version_0/PPO_.../checkpoint_000010'); "
             "recorded in the version's reinforcement_learning_configuration.yaml"
    )

    arguments = parser.parse_args()
    configuration_class = get_configuration_class(arguments.experimentation_configuration_file)
    if arguments.initialization_checkpoint_path is not None:
        configuration_class.reinforcement_learning_configuration.initialization_checkpoint_path = Path(
            arguments.initialization_checkpoint_path
        )

    reinforcement_learning_training(configuration_class)
