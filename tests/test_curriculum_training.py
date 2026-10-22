"""Curriculum training on CartPole: stages chained from the previous weights, and resumed after an interruption.

They start Ray and train for real (a few minutes in total).
"""

import json
import pickle
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import numpy as np
import pytest
import torch

from maxalaar_reinforcement_learning.configuration.curriculum_configuration import CurriculumConfiguration
from maxalaar_reinforcement_learning.configuration.experimentation_configuration import ExperimentationConfiguration
from maxalaar_reinforcement_learning.training.curriculum_training import check_stages, curriculum_training

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GPUS = 1 if torch.cuda.is_available() else 0


def write_stage(directory: Path, name: str, stopping_criterion: dict, learning_rate: float = 3e-4) -> Path:
    """A small CartPole stage file (configuration files define a variable named like the file)."""
    path = directory / f"{name}.py"
    path.write_text(textwrap.dedent(f"""
        from maxalaar_reinforcement_learning.architectures.dense_ppo import DensePPO
        from maxalaar_reinforcement_learning.configuration.experimentation_configuration import ExperimentationConfiguration

        {name} = ExperimentationConfiguration(experimentation_name="{name}", environment_name="CartPoleRllib")
        configuration = {name}.reinforcement_learning_configuration
        configuration.architecture = DensePPO
        configuration.architecture_configuration = {{"configuration_hidden_layers": [32, 32]}}
        configuration.number_environment_runners = 1
        configuration.number_learners = 1
        configuration.number_gpus_per_learner = {GPUS}
        configuration.train_batch_size = 400
        configuration.minibatch_size = 100
        configuration.number_epochs = 1
        configuration.learning_rate = {learning_rate}
        configuration.evaluation_interval = None
        configuration.stopping_criterion = {stopping_criterion!r}
    """))
    return path


def module_weights(checkpoint: Path) -> dict:
    with open(checkpoint / "learner_group" / "learner" / "rl_module" / "default_policy" / "module_state.pkl", "rb") as file:
        return pickle.load(file)


def test_each_stage_starts_from_the_weights_of_the_previous_one(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    first = write_stage(tmp_path, "first_stage", {"training_iteration": 2})
    # A zero learning rate keeps the weights the stage starts from.
    second = write_stage(tmp_path, "second_stage", {"training_iteration": 1}, learning_rate=0.0)
    curriculum = CurriculumConfiguration("curriculum_test", stages=[first.name, second.name])

    assert curriculum_training(curriculum)

    state = json.loads((tmp_path / "experiments" / "curriculum_test" / "version_0" / "curriculum_state.json").read_text())
    assert [stage["status"] for stage in state["stages"]] == ["completed", "completed"]
    first_weights = module_weights(Path(state["stages"][0]["final_checkpoint_path"]))
    second_weights = module_weights(Path(state["stages"][1]["final_checkpoint_path"]))
    assert first_weights.keys() == second_weights.keys()
    assert all(np.allclose(first_weights[key], second_weights[key]) for key in first_weights)

    # Training a completed curriculum again does nothing.
    assert curriculum_training(curriculum)


def test_an_interrupted_stage_resumes_where_it_stopped(tmp_path: Path):
    write_stage(tmp_path, "first_stage", {"training_iteration": 1})
    write_stage(tmp_path, "second_stage", {"training_iteration": 6})
    (tmp_path / "interrupted_curriculum.py").write_text(textwrap.dedent("""
        from maxalaar_reinforcement_learning.configuration.curriculum_configuration import CurriculumConfiguration

        interrupted_curriculum = CurriculumConfiguration(
            "interrupted_curriculum", stages=["first_stage.py", "second_stage.py"]
        )
        interrupted_curriculum.checkpoint_frequency = 1
    """))
    command = [sys.executable, str(PROJECT_ROOT / "scripts" / "curriculum_training.py"),
               "--curriculum_configuration_file", "interrupted_curriculum.py"]
    version_path = tmp_path / "experiments" / "interrupted_curriculum" / "version_0"

    # Interrupted (Ctrl+C) once the second stage has a checkpoint.
    process = subprocess.Popen(command, cwd=tmp_path, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    deadline = time.time() + 300
    while not list(version_path.glob("stage_2_second_stage/**/checkpoint_*")):
        assert process.poll() is None and time.time() < deadline, "the second stage wrote no checkpoint"
        time.sleep(1)
    process.send_signal(signal.SIGINT)
    assert process.wait(timeout=300) != 0, "an interrupted curriculum is not completed"
    state = json.loads((version_path / "curriculum_state.json").read_text())
    assert [stage["status"] for stage in state["stages"]] == ["completed", "running"]

    resumed = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, timeout=600)
    assert resumed.returncode == 0, resumed.stderr[-3000:]
    log = (version_path / "curriculum.log").read_text()
    assert log.count("stage 1 (first_stage.py): started") == 1
    assert "stage 2 (second_stage.py): resumed" in log
    state = json.loads((version_path / "curriculum_state.json").read_text())
    assert [stage["status"] for stage in state["stages"]] == ["completed", "completed"]


def test_stages_must_end_and_share_their_architecture():
    def stage(stopping_criterion, hidden_layers=(32, 32)) -> ExperimentationConfiguration:
        configuration = ExperimentationConfiguration(experimentation_name="stage", environment_name="CartPoleRllib")
        configuration.reinforcement_learning_configuration.stopping_criterion = stopping_criterion
        configuration.reinforcement_learning_configuration.architecture_configuration = {
            "configuration_hidden_layers": list(hidden_layers)
        }
        return configuration

    curriculum = CurriculumConfiguration("check_test", stages=["first.py", "second.py"])
    check_stages(curriculum, [stage({"training_iteration": 1}), stage(None)])  # the last one may run forever
    with pytest.raises(ValueError, match="stopping criterion"):
        check_stages(curriculum, [stage(None), stage(None)])
    with pytest.raises(ValueError, match="architecture_configuration"):
        check_stages(curriculum, [stage({"training_iteration": 1}), stage(None, hidden_layers=(64,))])
