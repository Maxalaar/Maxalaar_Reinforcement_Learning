import argparse
from pathlib import Path

from maxalaar_reinforcement_learning.configuration.experimentation_configuration import ExperimentationConfiguration
from maxalaar_reinforcement_learning.environments.register_environments import register_environments
from maxalaar_reinforcement_learning.training.run_training import resume_training
from maxalaar_reinforcement_learning.utilities.initialize_ray import initialize_ray
from maxalaar_reinforcement_learning.utilities.save.get_configuration_class import get_configuration_class


def continue_reinforcement_learning_training(
        experimentation_configuration: ExperimentationConfiguration,
        reinforcement_learning_path: Path,
):
    initialize_ray(experimentation_configuration)
    register_environments()
    resume_training(experimentation_configuration, reinforcement_learning_path)


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
