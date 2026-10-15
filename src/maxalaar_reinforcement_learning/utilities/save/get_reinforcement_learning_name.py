from pathlib import Path

from maxalaar_reinforcement_learning.configuration.experimentation_configuration import ExperimentationConfiguration


def get_reinforcement_learning_name(
    experimentation_configuration: ExperimentationConfiguration,
    reinforcement_learning_path: Path,
) -> Path:

    surrogate_policy_name = reinforcement_learning_path.relative_to(experimentation_configuration.reinforcement_learning_storage_path)

    return surrogate_policy_name