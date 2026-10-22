import argparse

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

    arguments = parser.parse_args()
    configuration_class = get_configuration_class(arguments.experimentation_configuration_file)

    reinforcement_learning_training(configuration_class)
