from typing import List, NamedTuple, Optional, Tuple
from datetime import datetime
from pathlib import Path

from ray._private.dict import unflattened_lookup
from ray.train import Checkpoint
from ray.train._internal.session import _TrainingResult
from ray.tune import ExperimentAnalysis
from ray.tune.experiment import Trial

from maxalaar_reinforcement_learning.configuration.experimentation_configuration import ExperimentationConfiguration


def resolve_checkpoint_score_criterion(
        experimentation_configuration: ExperimentationConfiguration,
) -> Tuple[str, str]:
    """The (metric, mode) pair used to rank checkpoints, taken from the display
    episodes configuration when set, otherwise from the reinforcement learning one."""
    display_episodes_configuration = experimentation_configuration.display_episodes_configuration
    if display_episodes_configuration.checkpoint_score_attribute is None:
        reinforcement_learning_configuration = experimentation_configuration.reinforcement_learning_configuration
        return (
            reinforcement_learning_configuration.checkpoint_score_attribute,
            reinforcement_learning_configuration.checkpoint_score_order,
        )
    return (
        display_episodes_configuration.checkpoint_score_attribute,
        display_episodes_configuration.checkpoint_score_order,
    )


def find_best_reinforcement_learning_checkpoint_path(
        metric: str,
        mode: str,
        reinforcement_learning_path: Path,
):
    analysis = ExperimentAnalysis(reinforcement_learning_path)

    best_trial: Trial = analysis.get_best_trial(
        metric=metric,
        mode=mode,
    )

    best_checkpoint: Checkpoint = analysis.get_best_checkpoint(
        trial=best_trial,
        metric=metric,
        mode=mode,
    )
    best_checkpoint_metric: Optional[dict] = None

    if best_checkpoint is None:
        raise ValueError("Error: 'best_checkpoint' is None. Ray couldn't find any checkpoint.")

    training_results: List[_TrainingResult] = best_trial.run_metadata.checkpoint_manager.best_checkpoint_results
    for training_result in training_results:
        if training_result.checkpoint.path == best_checkpoint.path:
            best_checkpoint_metric = training_result.metrics
            break

    creation_time = datetime.strptime(best_checkpoint_metric['date'], '%Y-%m-%d_%H-%M-%S')

    print('Best checkpoint is create on: ' + creation_time.strftime("On %B %d, %Y at %H:%M and %S seconds."))
    print('Metric use : ' + str(metric))
    print('Mode use : ' + str(mode))
    print('Best checkpoint metric value : ' + str(unflattened_lookup(metric, best_checkpoint_metric)))
    print('Best checkpoint path : ' + str(best_checkpoint.path))
    print()

    return Path(best_checkpoint.path)


class ReinforcementLearningCheckpoint(NamedTuple):
    path: Path
    metric_value: Optional[float]
    creation_time: Optional[datetime]


def find_all_reinforcement_learning_checkpoints(
        metric: str,
        reinforcement_learning_path: Path,
) -> List[ReinforcementLearningCheckpoint]:
    """Every retained checkpoint stored under `reinforcement_learning_path`, each
    paired with its value for `metric` and its creation time (`metrics['date']`).

    The list is returned unsorted: the caller decides how to rank the checkpoints
    (by metric value, or by creation time when runs are chained through
    `initialization_checkpoint_path` and a per-run metric would reset).
    """
    analysis = ExperimentAnalysis(reinforcement_learning_path)

    checkpoints: List[ReinforcementLearningCheckpoint] = []
    for trial in analysis.trials:
        training_results: List[_TrainingResult] = trial.run_metadata.checkpoint_manager.best_checkpoint_results
        for training_result in training_results:
            metric_value = unflattened_lookup(metric, training_result.metrics)

            raw_date = training_result.metrics.get('date')
            creation_time = datetime.strptime(raw_date, '%Y-%m-%d_%H-%M-%S') if raw_date else None

            checkpoints.append(ReinforcementLearningCheckpoint(
                path=Path(training_result.checkpoint.path),
                metric_value=None if metric_value is None else float(metric_value),
                creation_time=creation_time,
            ))

    return checkpoints
