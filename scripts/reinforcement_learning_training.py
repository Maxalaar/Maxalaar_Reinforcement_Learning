import argparse

from ray import tune
from ray.tune import RunConfig, CheckpointConfig
from ray.air.integrations.wandb import WandbLoggerCallback

from maxalaar_reinforcement_learning.configuration.experimentation_configuration import ExperimentationConfiguration
from maxalaar_reinforcement_learning.environments.register_environments import register_environments
from maxalaar_reinforcement_learning.utilities.check_wandb_api_key_file import check_wandb_api_key_file
from maxalaar_reinforcement_learning.utilities.initialize_ray import initialize_ray
from maxalaar_reinforcement_learning.training.build_algorithm_configuration import build_algorithm_configuration
from maxalaar_reinforcement_learning.utilities.save.get_reinforcement_learning_version_number import get_reinforcement_learning_version_number
from maxalaar_reinforcement_learning.utilities.save.get_configuration_class import get_configuration_class


def reinforcement_learning_training(
        experimentation_configuration: ExperimentationConfiguration,
):
    initialize_ray(experimentation_configuration)
    register_environments()
    reinforcement_learning_configuration = experimentation_configuration.reinforcement_learning_configuration

    algorithm, algorithm_configuration = build_algorithm_configuration(experimentation_configuration)

    storage_path = experimentation_configuration.reinforcement_learning_storage_path / reinforcement_learning_configuration.training_name
    version_name = 'version_' + str(get_reinforcement_learning_version_number(storage_path))
    reinforcement_learning_configuration.to_yaml_file(storage_path / version_name)
    experimentation_configuration.environment_to_yaml_file(storage_path / version_name)

    callbacks = []
    if reinforcement_learning_configuration.use_wandb:
        check_wandb_api_key_file(experimentation_configuration.wandb_api_key_file_path)
        callbacks.append(
            WandbLoggerCallback(
                project='Clustering_Agent_Latent_Space',
                api_key_file=str(experimentation_configuration.wandb_api_key_file_path),
                log_config=True,
            )
        )

    tuner = tune.Tuner(
        trainable=algorithm,
        param_space=algorithm_configuration,
        run_config=RunConfig(
            name=version_name,
            storage_path=str(storage_path),
            stop=reinforcement_learning_configuration.stopping_criterion,
            checkpoint_config=CheckpointConfig(
                num_to_keep=reinforcement_learning_configuration.number_checkpoint_to_keep,
                checkpoint_score_attribute=reinforcement_learning_configuration.checkpoint_score_attribute,
                checkpoint_score_order=reinforcement_learning_configuration.checkpoint_score_order,
                checkpoint_frequency=reinforcement_learning_configuration.checkpoint_frequency,
                checkpoint_at_end=True,
            ),
            callbacks=callbacks,
        ),
    )

    tuner.fit()


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
