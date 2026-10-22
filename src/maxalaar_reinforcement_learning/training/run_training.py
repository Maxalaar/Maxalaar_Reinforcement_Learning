"""Start or resume one reinforcement learning training (one Tune trial), and report how it ended.

Ray must be initialized by the caller (utilities/initialize_ray.py), and the environments registered.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ray import tune
from ray.air.integrations.wandb import WandbLoggerCallback
from ray.tune import CheckpointConfig, RunConfig

from maxalaar_reinforcement_learning.configuration.experimentation_configuration import ExperimentationConfiguration
from maxalaar_reinforcement_learning.training.build_algorithm_configuration import build_algorithm_configuration
from maxalaar_reinforcement_learning.utilities.check_wandb_api_key_file import check_wandb_api_key_file
from maxalaar_reinforcement_learning.utilities.save.get_reinforcement_learning_version_number import get_reinforcement_learning_version_number


@dataclass
class TrainingOutcome:
    version_path: Path
    # Last reported result (flattened keys, e.g. 'env_runners/episode_return_mean'), None when nothing was reported.
    last_result: Optional[dict]
    # Last checkpoint of the trial: the final one when the stopping criterion was reached (checkpoint_at_end).
    last_checkpoint_path: Optional[Path]
    error: Optional[BaseException]


def create_training_version(experimentation_configuration: ExperimentationConfiguration) -> Path:
    """A new version directory of the training, with the configurations written in it."""
    reinforcement_learning_configuration = experimentation_configuration.reinforcement_learning_configuration
    storage_path = experimentation_configuration.reinforcement_learning_storage_path / reinforcement_learning_configuration.training_name
    version_path = storage_path / ('version_' + str(get_reinforcement_learning_version_number(storage_path)))
    reinforcement_learning_configuration.to_yaml_file(version_path)
    experimentation_configuration.environment_to_yaml_file(version_path)
    return version_path


def start_training(experimentation_configuration: ExperimentationConfiguration, version_path: Path) -> TrainingOutcome:
    algorithm, algorithm_configuration = build_algorithm_configuration(experimentation_configuration)
    tuner = tune.Tuner(
        trainable=algorithm,
        param_space=algorithm_configuration,
        run_config=_run_configuration(experimentation_configuration, version_path),
    )
    return _fit(tuner, version_path)


def resume_training(experimentation_configuration: ExperimentationConfiguration, version_path: Path) -> TrainingOutcome:
    """Resume the trial of ``version_path`` from its last checkpoint (it must hold the tuner.pkl Tune writes)."""
    if not (version_path / 'tuner.pkl').exists():
        raise FileNotFoundError(
            f"Tuner checkpoint not found at: {version_path / 'tuner.pkl'}. "
            f"Make sure the path points to a version directory (e.g., .../version_0/) "
            f"that contains a tuner.pkl file."
        )

    # Rebuild the algorithm configuration from the (possibly updated) experimentation
    # configuration file, so config changes made since the run started (e.g. a new
    # callback) are picked up on resume instead of silently reusing the pickled config.
    algorithm, algorithm_configuration = build_algorithm_configuration(experimentation_configuration)
    tuner = tune.Tuner.restore(
        str(version_path),
        trainable=algorithm,
        param_space=algorithm_configuration,
        resume_unfinished=True,
        resume_errored=True,
        restart_errored=True,
    )
    return _fit(tuner, version_path)


def stopping_criterion_reached(stopping_criterion: Optional[dict], result: Optional[dict]) -> bool:
    """Whether ``result`` meets every value of the (Tune) stopping criterion; never for a training without one."""
    if not stopping_criterion or not result:
        return False
    return all(key in result and result[key] >= value for key, value in stopping_criterion.items())


def _run_configuration(experimentation_configuration: ExperimentationConfiguration, version_path: Path) -> RunConfig:
    reinforcement_learning_configuration = experimentation_configuration.reinforcement_learning_configuration

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

    return RunConfig(
        name=version_path.name,
        storage_path=str(version_path.parent),
        stop=reinforcement_learning_configuration.stopping_criterion,
        checkpoint_config=CheckpointConfig(
            num_to_keep=reinforcement_learning_configuration.number_checkpoint_to_keep,
            checkpoint_score_attribute=reinforcement_learning_configuration.checkpoint_score_attribute,
            checkpoint_score_order=reinforcement_learning_configuration.checkpoint_score_order,
            checkpoint_frequency=reinforcement_learning_configuration.checkpoint_frequency,
            checkpoint_at_end=True,
        ),
        callbacks=callbacks,
    )


def _fit(tuner: tune.Tuner, version_path: Path) -> TrainingOutcome:
    result = tuner.fit()[0]
    return TrainingOutcome(
        version_path=version_path,
        last_result=_flattened(result.metrics) if result.metrics else None,
        last_checkpoint_path=Path(result.checkpoint.path) if result.checkpoint is not None else None,
        error=result.error,
    )


def _flattened(dictionary: dict, prefix: str = '') -> dict:
    flattened = {}
    for key, value in dictionary.items():
        if isinstance(value, dict):
            flattened.update(_flattened(value, prefix + key + '/'))
        else:
            flattened[prefix + key] = value
    return flattened
