"""Train the stages of a curriculum one after the other, and resume it where it stopped.

experiments/<curriculum name>/version_<k>/
    curriculum_state.json   status of each stage (pending, running, completed), its training and final checkpoint
    curriculum.log          one line per event
    stage_<i>_<configuration file name>/reinforcement_learning/...   the usual training directories of the stage
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import ray

from maxalaar_reinforcement_learning.configuration.curriculum_configuration import CurriculumConfiguration
from maxalaar_reinforcement_learning.configuration.experimentation_configuration import ExperimentationConfiguration
from maxalaar_reinforcement_learning.environments.register_environments import register_environments
from maxalaar_reinforcement_learning.training.run_training import (
    TrainingOutcome,
    create_training_version,
    resume_training,
    start_training,
    stopping_criterion_reached,
)
from maxalaar_reinforcement_learning.utilities.initialize_ray import initialize_ray
from maxalaar_reinforcement_learning.utilities.save.get_configuration_class import get_configuration_class
from maxalaar_reinforcement_learning.utilities.save.get_reinforcement_learning_version_number import get_reinforcement_learning_version_number

STATE_FILE_NAME = 'curriculum_state.json'
LOG_FILE_NAME = 'curriculum.log'


def curriculum_training(curriculum_configuration: CurriculumConfiguration, new: bool = False) -> bool:
    """Train the curriculum's stages that are not completed yet, in order; True when every stage is completed.

    It resumes the latest run of the curriculum (``new``: starts a new one). It stops at the first stage that ends
    without reaching its stopping criterion (interrupted, or failed): training it again resumes that stage.
    """
    stages = [get_configuration_class(str(stage)) for stage in curriculum_configuration.stages]
    check_stages(curriculum_configuration, stages)

    version_path = _curriculum_version_path(curriculum_configuration, new)
    state = _load_state(version_path, curriculum_configuration)
    _log(version_path, f'curriculum {curriculum_configuration.curriculum_name}, {version_path.name}')

    for index, (stage_file, stage) in enumerate(zip(curriculum_configuration.stages, stages)):
        stage_state = state['stages'][index]
        if stage_state['status'] == 'completed':
            continue

        previous_checkpoint = Path(state['stages'][index - 1]['final_checkpoint_path']) if index > 0 else None
        _prepare_stage(stage, version_path / f'stage_{index}_{stage_file.stem}', curriculum_configuration, previous_checkpoint)

        initialize_ray(stage)
        try:
            register_environments()
            training_path = Path(stage_state['training_path']) if stage_state['training_path'] else None
            if training_path is not None and (training_path / 'tuner.pkl').exists():
                _log(version_path, f'stage {index} ({stage_file}): resumed from {training_path}')
                outcome = resume_training(stage, training_path)
            else:
                training_path = create_training_version(stage)
                stage_state.update(status='running', training_path=str(training_path), started=_now())
                _save_state(version_path, state)
                _log(version_path, f'stage {index} ({stage_file}): started in {training_path}'
                     + (f', from {previous_checkpoint}' if previous_checkpoint else ''))
                outcome = start_training(stage, training_path)
        finally:
            ray.shutdown()

        criterion = stage.reinforcement_learning_configuration.stopping_criterion
        if not _stage_completed(criterion, outcome):
            _log(version_path, f'stage {index} ({stage_file}): ended without reaching {criterion}'
                 + (f' ({outcome.error!r})' if outcome.error else '') + '; training the curriculum again resumes it')
            return False

        stage_state.update(status='completed', final_checkpoint_path=str(outcome.last_checkpoint_path), completed=_now())
        _save_state(version_path, state)
        _log(version_path, f'stage {index} ({stage_file}): completed, final checkpoint {outcome.last_checkpoint_path}')

    _log(version_path, 'curriculum completed')
    return True


def check_stages(curriculum_configuration: CurriculumConfiguration, stages: list[ExperimentationConfiguration]) -> None:
    """Every stage but the last needs a stopping criterion, and the weights of a stage must fit the next one."""
    for stage_file, stage in zip(curriculum_configuration.stages[:-1], stages[:-1]):
        if not stage.reinforcement_learning_configuration.stopping_criterion:
            raise ValueError(f'{stage_file}: every stage but the last needs a stopping criterion, to end')

    first = stages[0].reinforcement_learning_configuration
    for stage_file, stage in zip(curriculum_configuration.stages[1:], stages[1:]):
        configuration = stage.reinforcement_learning_configuration
        for field in ('architecture', 'architecture_configuration', 'policies', 'algorithm_name'):
            if getattr(configuration, field) != getattr(first, field):
                raise ValueError(
                    f'{stage_file}: its {field} differs from the first stage ({curriculum_configuration.stages[0]}), '
                    f'so it cannot start from the weights of the previous stage'
                )


def _prepare_stage(
        stage: ExperimentationConfiguration,
        storage_path: Path,
        curriculum_configuration: CurriculumConfiguration,
        previous_checkpoint: Optional[Path],
) -> None:
    stage.experimentation_storage_path = storage_path
    stage.videos_path = storage_path / 'videos'
    stage.video_episodes_storage_path = stage.videos_path / 'episodes'
    stage.reinforcement_learning_storage_path = storage_path / 'reinforcement_learning'
    reinforcement_learning_configuration = stage.reinforcement_learning_configuration
    reinforcement_learning_configuration.checkpoint_frequency = curriculum_configuration.checkpoint_frequency
    if previous_checkpoint is not None:
        # Weights only: the optimizer of each stage starts anew.
        reinforcement_learning_configuration.initialization_checkpoint_path = previous_checkpoint


def _stage_completed(criterion: Optional[dict], outcome: TrainingOutcome) -> bool:
    return (
        outcome.error is None
        and outcome.last_checkpoint_path is not None
        and stopping_criterion_reached(criterion, outcome.last_result)
    )


def _curriculum_version_path(curriculum_configuration: CurriculumConfiguration, new: bool) -> Path:
    storage_path = curriculum_configuration.curriculum_storage_path
    number = get_reinforcement_learning_version_number(storage_path)
    if not new and number > 0:
        number -= 1  # the latest run
    version_path = storage_path / f'version_{number}'
    version_path.mkdir(parents=True, exist_ok=True)
    return version_path


def _load_state(version_path: Path, curriculum_configuration: CurriculumConfiguration) -> dict:
    stage_files = [str(stage) for stage in curriculum_configuration.stages]
    state_path = version_path / STATE_FILE_NAME
    if not state_path.exists():
        state = {
            'curriculum_name': curriculum_configuration.curriculum_name,
            'stages': [
                {'configuration_file': stage_file, 'status': 'pending', 'training_path': None,
                 'final_checkpoint_path': None, 'started': None, 'completed': None}
                for stage_file in stage_files
            ],
        }
        _save_state(version_path, state)
        return state

    state = json.loads(state_path.read_text())
    if [stage['configuration_file'] for stage in state['stages']] != stage_files:
        raise ValueError(
            f'{state_path} was made for other stages: {[stage["configuration_file"] for stage in state["stages"]]}; '
            f'train a new run of the curriculum (new=True, --new) to change them'
        )
    return state


def _save_state(version_path: Path, state: dict) -> None:
    temporary_path = version_path / (STATE_FILE_NAME + '.tmp')
    temporary_path.write_text(json.dumps(state, indent=4))
    temporary_path.replace(version_path / STATE_FILE_NAME)  # atomic: an interruption never leaves half a state


def _log(version_path: Path, message: str) -> None:
    line = f'{_now()} {message}'
    print('[curriculum] ' + line)
    with open(version_path / LOG_FILE_NAME, 'a') as file:
        file.write(line + '\n')


def _now() -> str:
    return datetime.now().isoformat(timespec='seconds')
