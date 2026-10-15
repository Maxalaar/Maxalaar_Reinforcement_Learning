"""End-to-end checks on CartPole: training, checkpoint restore (optimizer state), export.

They start Ray and train for real (about a minute in total); the learner uses the GPU when there is one,
which is where restoring the optimizer state used to crash (see patches/).
"""

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import ray
import torch

from maxalaar_reinforcement_learning.configuration.experimentation_configuration import ExperimentationConfiguration
from maxalaar_reinforcement_learning.environments.register_environments import register_environments
from maxalaar_reinforcement_learning.training.build_algorithm_configuration import build_algorithm_configuration
from maxalaar_reinforcement_learning.utilities.initialize_ray import initialize_ray

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GPUS = 1 if torch.cuda.is_available() else 0

# Small and fast settings, written into the configuration files of the script test too. As in real trainings, the
# learner and the environment runner are separate Ray actors (the learner on the GPU). The architecture is
# one of this package: the ONNX export relies on its forward_onnx (RLlib's default module has none).
SMALL_SETTINGS = f"""
from maxalaar_reinforcement_learning.architectures.dense_ppo import DensePPO
configuration.reinforcement_learning_configuration.architecture = DensePPO
configuration.reinforcement_learning_configuration.architecture_configuration = {{'configuration_hidden_layers': [32, 32]}}
configuration.reinforcement_learning_configuration.number_environment_runners = 1
configuration.reinforcement_learning_configuration.number_learners = 1
configuration.reinforcement_learning_configuration.number_gpus_per_learner = {GPUS}
configuration.reinforcement_learning_configuration.train_batch_size = 400
configuration.reinforcement_learning_configuration.minibatch_size = 100
configuration.reinforcement_learning_configuration.number_epochs = 1
configuration.reinforcement_learning_configuration.evaluation_interval = 1
configuration.reinforcement_learning_configuration.evaluation_num_environment_runners = 1
configuration.reinforcement_learning_configuration.evaluation_duration = 2
configuration.reinforcement_learning_configuration.checkpoint_frequency = 1
configuration.reinforcement_learning_configuration.stopping_criterion = {{'training_iteration': 2}}
"""


def small_configuration(name: str) -> ExperimentationConfiguration:
    namespace = {"configuration": ExperimentationConfiguration(experimentation_name=name, environment_name="CartPoleRllib")}
    exec(SMALL_SETTINGS, namespace)
    return namespace["configuration"]


def train_save_restore_train(configuration: ExperimentationConfiguration, checkpoint: Path) -> dict:
    initialize_ray(configuration)
    try:
        register_environments()
        _, algorithm_configuration = build_algorithm_configuration(configuration)
        algorithm = algorithm_configuration.build_algo()
        algorithm.train()
        algorithm.save_to_path(str(checkpoint))
        algorithm.stop()

        restored = algorithm_configuration.build_algo()
        restored.restore_from_path(str(checkpoint))
        result = restored.train()  # steps the restored optimizer: crashed on GPU without the patch
        restored.stop()
        return result
    finally:
        ray.shutdown()


def test_training_resumes_from_a_checkpoint(tmp_path: Path):
    result = train_save_restore_train(small_configuration("restore_test"), tmp_path / "checkpoint")
    assert result["training_iteration"] == 2


@pytest.mark.xfail(
    strict=True,
    reason="Known bug, already in the thesis repository with Ray 2.48: after a restore, DensePPO evaluated on the "
    "local environment runner (evaluation_num_environment_runners = 0) gets inputs on the GPU and weights on the CPU.",
)
def test_dense_ppo_resumes_with_evaluation_on_the_local_runner(tmp_path: Path):
    configuration = small_configuration("restore_local_evaluation_test")
    configuration.reinforcement_learning_configuration.evaluation_num_environment_runners = 0
    train_save_restore_train(configuration, tmp_path / "checkpoint")


def run_script(name: str, *arguments: str, working_directory: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / name), *arguments],
        cwd=working_directory,
        capture_output=True,
        text=True,
        timeout=600,
    )


def test_training_script_then_export(tmp_path: Path):
    # Configuration files define a variable named like the file.
    configuration_file = tmp_path / "script_test.py"
    configuration_file.write_text(
        textwrap.dedent("""
            from maxalaar_reinforcement_learning.configuration.experimentation_configuration import ExperimentationConfiguration

            configuration = ExperimentationConfiguration(experimentation_name='script_test', environment_name='CartPoleRllib')
        """)
        + SMALL_SETTINGS
        + "script_test = configuration\n"
    )

    training = run_script(
        "reinforcement_learning_training.py",
        "--experimentation_configuration_file",
        str(configuration_file),
        working_directory=tmp_path,
    )
    assert training.returncode == 0, training.stderr[-3000:]
    checkpoints = list((tmp_path / "experiments" / "script_test").rglob("checkpoint_*"))
    assert checkpoints, "the training wrote no checkpoint"

    export = run_script(
        "export_reinforcement_learning_module.py",
        "--checkpoints_source",
        str(tmp_path / "experiments" / "script_test"),
        "--export_formats",
        "onnx",
        working_directory=tmp_path,
    )
    assert export.returncode == 0, export.stderr[-3000:]
    assert list((tmp_path / "experiments").rglob("*.onnx")), "the export wrote no ONNX file"


@pytest.mark.parametrize("path", sorted((PROJECT_ROOT / "configurations").rglob("*.py")), ids=lambda path: path.stem)
def test_experiment_configurations_load(path: Path):
    from maxalaar_reinforcement_learning.utilities.save.get_configuration_class import get_configuration_class

    if "cubeball" in path.parts:
        pytest.importorskip("cubeball")
    assert isinstance(get_configuration_class(str(path)), ExperimentationConfiguration)
