from pathlib import Path
from typing import List, NamedTuple, Tuple

import yaml

from maxalaar_reinforcement_learning.checkpoints.find_best_checkpoint_path import (
    ReinforcementLearningCheckpoint,
    find_all_reinforcement_learning_checkpoints,
)

# Weakest agent -> v1, strongest -> vN.
#   'creation_time': robust when runs are chained (a per-run metric resets).
#   'metric':        the checkpoint score criterion of each run.
ORDER_BY_CHOICES = ('creation_time', 'metric')

#   'best': only vN.  'all': every checkpoint.
SELECTION_CHOICES = ('best', 'all')

RUN_CONFIGURATION_FILE_NAME = 'reinforcement_learning_configuration.yaml'


class ScoreCriterion(NamedTuple):
    metric: str
    mode: str


class SelectedCheckpoint(NamedTuple):
    version_name: str
    checkpoint: ReinforcementLearningCheckpoint
    score_criterion: ScoreCriterion


class _IgnorePythonTagsLoader(yaml.SafeLoader):
    pass


# The run configuration holds `!!python/name:` tags: read them as None.
_IgnorePythonTagsLoader.add_multi_constructor('', lambda loader, tag_suffix, node: None)


def _read_score_criterion(experiment_analysis_directory: Path) -> ScoreCriterion:
    run_configuration_path = experiment_analysis_directory / RUN_CONFIGURATION_FILE_NAME
    if not run_configuration_path.is_file():
        raise ValueError('No ' + RUN_CONFIGURATION_FILE_NAME + ' in ' + str(experiment_analysis_directory))

    with open(run_configuration_path) as run_configuration_file:
        run_configuration = yaml.load(run_configuration_file, Loader=_IgnorePythonTagsLoader)

    metric = run_configuration.get('checkpoint_score_attribute')
    if metric is None:
        raise ValueError('No checkpoint_score_attribute in ' + str(run_configuration_path))
    return ScoreCriterion(metric=metric, mode=run_configuration.get('checkpoint_score_order', 'max'))


def _find_experiment_analysis_directories(source: Path) -> List[Path]:
    directories = set()
    for experiment_state_file in source.resolve().rglob('experiment_state-*.json'):
        directories.add(experiment_state_file.parent)
    return sorted(directories)


def _collect_checkpoints(
    checkpoints_sources: List[Path],
) -> List[Tuple[ReinforcementLearningCheckpoint, ScoreCriterion]]:
    collected: List[Tuple[ReinforcementLearningCheckpoint, ScoreCriterion]] = []
    seen_checkpoint_paths = set()

    experiment_analysis_directories = set()
    for source in checkpoints_sources:
        experiment_analysis_directories.update(_find_experiment_analysis_directories(source))

    for experiment_analysis_directory in sorted(experiment_analysis_directories):
        print('Scanning: ' + str(experiment_analysis_directory))
        score_criterion = _read_score_criterion(experiment_analysis_directory)
        for checkpoint in find_all_reinforcement_learning_checkpoints(score_criterion.metric, experiment_analysis_directory):
            if checkpoint.path in seen_checkpoint_paths:
                continue
            seen_checkpoint_paths.add(checkpoint.path)
            collected.append((checkpoint, score_criterion))

    return collected


def _rank_checkpoints(
    checkpoints: List[Tuple[ReinforcementLearningCheckpoint, ScoreCriterion]],
    order_by: str,
) -> List[Tuple[ReinforcementLearningCheckpoint, ScoreCriterion]]:
    if order_by == 'creation_time':
        missing = [checkpoint for checkpoint, _ in checkpoints if checkpoint.creation_time is None]
        if missing:
            raise ValueError(
                'Cannot order by creation_time, some checkpoints have no date: '
                + ', '.join(str(checkpoint.path) for checkpoint in missing)
            )
        return sorted(checkpoints, key=lambda item: item[0].creation_time)

    score_criteria = {score_criterion for _, score_criterion in checkpoints}
    if len(score_criteria) > 1:
        raise ValueError(
            'Cannot order by metric, the runs were trained with different score criteria: '
            + ', '.join(str(score_criterion) for score_criterion in sorted(score_criteria))
        )
    mode = score_criteria.pop().mode

    missing = [checkpoint for checkpoint, _ in checkpoints if checkpoint.metric_value is None]
    if missing:
        raise ValueError(
            'Cannot order by metric, some checkpoints have no value for it: '
            + ', '.join(str(checkpoint.path) for checkpoint in missing)
        )
    return sorted(checkpoints, key=lambda item: item[0].metric_value, reverse=(mode == 'min'))


def select_checkpoints(
    checkpoints_sources: List[Path],
    selection: str = 'best',
    order_by: str = 'creation_time',
) -> Tuple[List[SelectedCheckpoint], List[Path]]:
    """The checkpoints to export, and the paths of every checkpoint found."""
    checkpoints = _collect_checkpoints(checkpoints_sources)
    if len(checkpoints) == 0:
        raise ValueError(
            'No checkpoint found under: ' + ', '.join(str(source) for source in checkpoints_sources)
        )

    selected = [
        SelectedCheckpoint('v' + str(version_index), checkpoint, score_criterion)
        for version_index, (checkpoint, score_criterion) in enumerate(_rank_checkpoints(checkpoints, order_by), start=1)
    ]
    if selection == 'best':
        selected = selected[-1:]

    return selected, [checkpoint.path for checkpoint, _ in checkpoints]
