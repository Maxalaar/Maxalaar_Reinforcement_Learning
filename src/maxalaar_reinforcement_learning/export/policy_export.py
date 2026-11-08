"""A policy exported to be played without its code: the network in ONNX and a description of what it expects.

An RLlib checkpoint needs the module's class, under its import path, to load: renaming, moving or changing the class
breaks every checkpoint made before. The exported policy needs none of it:

    <checkpoint>/policies/<policy>/policy.onnx   the actor: observation -> action distribution inputs (RLlib's
                                                 ACTION_DIST_INPUTS, from forward_inference); free batch size
    <checkpoint>/policies/<policy>/policy.json   what goes with it: the ONNX inputs (one per leaf of the observation
                                                 space), the spaces, the action distribution, the environment
                                                 configuration, the module's class and the commits of the code

The action is drawn from the distribution inputs by the code that plays (a game's own code), and the observation is
built by it too: the environment configuration of policy.json says which one the policy was trained on. Recurrent
modules are not supported.
"""

import dataclasses
import enum
import json
import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import gymnasium as gym
import numpy as np
import torch
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module import RLModule
from ray.rllib.utils.serialization import gym_space_to_dict
from ray.tune import Callback

from maxalaar_reinforcement_learning.export.export_reinforcement_learning_module import _load_policy_modules

POLICIES_DIRECTORY_NAME = 'policies'
ONNX_FILE_NAME = 'policy.onnx'
DESCRIPTION_FILE_NAME = 'policy.json'
OUTPUT_NAME = 'action_distribution_inputs'
# The input of a policy whose observation space is not a Dict.
OBSERVATION_INPUT_NAME = 'observation'

logger = logging.getLogger(__name__)


def observation_paths(space: gym.Space) -> list[tuple[str, ...]]:
    """The path of every leaf of an observation space, in the order of the ONNX inputs (Dict keys sorted)."""
    if isinstance(space, gym.spaces.Dict):
        return [(key, *path) for key in sorted(space.spaces) for path in observation_paths(space.spaces[key])]
    return [()]


def input_name(path: tuple[str, ...]) -> str:
    return '.'.join(path) or OBSERVATION_INPUT_NAME


def _leaf(space: gym.Space, path: tuple[str, ...]) -> gym.Space:
    for key in path:
        space = space.spaces[key]
    return space


def _nest(paths: list[tuple[str, ...]], tensors) -> Any:
    if paths == [()]:
        return tensors[0]
    observation: dict = {}
    for path, tensor in zip(paths, tensors):
        node = observation
        for key in path[:-1]:
            node = node.setdefault(key, {})
        node[path[-1]] = tensor
    return observation


class _ActionDistributionInputs(torch.nn.Module):
    """The module's forward_inference, with the observation's leaves as separate inputs."""

    def __init__(self, module: RLModule, paths: list[tuple[str, ...]]):
        super().__init__()
        self.module = module
        self.paths = paths

    def forward(self, *leaves: torch.Tensor) -> torch.Tensor:
        return self.module.forward_inference({Columns.OBS: _nest(self.paths, leaves)})[Columns.ACTION_DIST_INPUTS]


def sample_observations(space: gym.Space, batch_size: int, seed: int) -> dict[str, np.ndarray]:
    """``batch_size`` observations drawn from ``space``, by ONNX input name."""
    space.seed(seed)
    samples = [space.sample() for _ in range(batch_size)]
    observations = {}
    for path in observation_paths(space):
        leaves = []
        for sample in samples:
            for key in path:
                sample = sample[key]
            leaves.append(np.asarray(sample, dtype=_leaf(space, path).dtype))
        observations[input_name(path)] = np.stack(leaves)
    return observations


def _check_onnx_outputs(onnx_file_path: Path, network: torch.nn.Module, space: gym.Space) -> None:
    """The ONNX outputs equal PyTorch's: the same infinite entries (masked actions), the finite ones within 1e-3 of the
    largest output."""
    import onnxruntime

    session = onnxruntime.InferenceSession(str(onnx_file_path), providers=['CPUExecutionProvider'])
    paths = observation_paths(space)
    for batch_size, seed in ((1, 1), (33, 2)):
        observations = sample_observations(space, batch_size, seed)
        with torch.no_grad():
            expected = network(*(torch.from_numpy(observations[input_name(path)]) for path in paths)).numpy()
        obtained = session.run([OUTPUT_NAME], observations)[0]
        finite = np.isfinite(expected)
        tolerance = 1e-3 * max(1.0, float(np.abs(expected[finite]).max(initial=0.0)))
        if not np.array_equal(finite, np.isfinite(obtained)) or np.abs(expected[finite] - obtained[finite]).max(
            initial=0.0
        ) > tolerance:
            raise ValueError(
                f'{onnx_file_path}: the ONNX outputs differ from PyTorch\'s (batch of {batch_size}, tolerance {tolerance:.1e})'
            )


def source_commits(paths: list[Path]) -> dict[str, dict]:
    """The git commit of each repository holding one of ``paths``, by repository root, and whether it had changes not
    committed (the exported policy may then come from code no commit has)."""
    commits = {}
    for path in paths:
        directory = path if path.is_dir() else path.parent

        def git(*arguments: str) -> str:
            return subprocess.run(
                ['git', '-C', str(directory), *arguments], capture_output=True, text=True, check=True
            ).stdout.strip()

        try:
            root = git('rev-parse', '--show-toplevel')
            if root not in commits:
                commits[root] = {'commit': git('rev-parse', 'HEAD'), 'uncommitted_changes': bool(git('status', '--porcelain'))}
        except (subprocess.CalledProcessError, FileNotFoundError):
            continue
    return commits


def to_json(value: Any) -> Any:
    """``value`` as JSON data, for reading only: a dataclass as its class and fields, a class as its import path,
    anything else unknown as its repr."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, enum.Enum):
        return to_json(value.value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, type):
        return value.__module__ + '.' + value.__qualname__
    if dataclasses.is_dataclass(value):
        return {
            'class': to_json(type(value)),
            **{field.name: to_json(getattr(value, field.name)) for field in dataclasses.fields(value)},
        }
    if isinstance(value, dict):
        return {str(key): to_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [to_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return repr(value)


def export_policy(module: RLModule, directory: Path, information: Optional[dict] = None) -> Path:
    """Write ``directory``/policy.onnx and policy.json for ``module`` (put on the CPU, in evaluation mode), the ONNX
    checked against PyTorch; ``information`` is added to policy.json as JSON data (to_json). Returns the directory."""
    if module.is_stateful():
        raise ValueError(f'{type(module).__qualname__}: the export of a recurrent policy is not supported')
    module = module.cpu().eval()
    # The architectures of this package keep the device their inputs go to (DensePPO: CUDA when there is a GPU).
    if isinstance(getattr(module, 'device', None), torch.device):
        module.device = torch.device('cpu')
    space = module.observation_space
    paths = observation_paths(space)
    network = _ActionDistributionInputs(module, paths).eval()

    directory.mkdir(parents=True, exist_ok=True)
    onnx_file_path = directory / ONNX_FILE_NAME
    example = sample_observations(space, batch_size=4, seed=0)
    batch_size = torch.export.Dim('batch_size', min=1, max=1 << 20)
    names = [input_name(path) for path in paths]
    with torch.no_grad():
        program = torch.onnx.export(
            network,
            tuple(torch.from_numpy(example[name]) for name in names),
            input_names=names,
            output_names=[OUTPUT_NAME],
            dynamic_shapes=(tuple({0: batch_size} for _ in names),),
            dynamo=True,
            verbose=False,
        )
    program.save(str(onnx_file_path))
    _check_onnx_outputs(onnx_file_path, network, space)

    module_class = type(module)
    module_file = Path(__import__(module_class.__module__, fromlist=['_']).__file__)
    description = {
        'onnx_file': ONNX_FILE_NAME,
        'inputs': [
            {
                'name': input_name(path),
                'observation_path': list(path),
                'shape': list(_leaf(space, path).shape),
                'dtype': str(_leaf(space, path).dtype),
            }
            for path in paths
        ],
        'output': {'name': OUTPUT_NAME, 'size': int(example_output_size(network, example, names))},
        'observation_space': gym_space_to_dict(space),
        'action_space': gym_space_to_dict(module.action_space),
        'action_distribution': to_json(module.get_inference_action_dist_cls()),
        'module_class': to_json(module_class),
        'source_commits': source_commits([Path.cwd(), Path(__file__), module_file]),
        'exported_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        **to_json(information or {}),
    }
    with open(directory / DESCRIPTION_FILE_NAME, 'w') as file:
        json.dump(description, file, indent=2)
    return directory


def example_output_size(network: torch.nn.Module, example: dict[str, np.ndarray], names: list[str]) -> int:
    with torch.no_grad():
        return network(*(torch.from_numpy(example[name]) for name in names)).shape[-1]


def export_checkpoint_policies(checkpoint_path: Path, information: Optional[dict] = None) -> list[Path]:
    """Export every policy of an algorithm checkpoint to <checkpoint>/policies/<policy>/."""
    checkpoint_path = Path(checkpoint_path).resolve()
    return [
        export_policy(
            module,
            checkpoint_path / POLICIES_DIRECTORY_NAME / policy_name,
            {**(information or {}), 'policy': policy_name, 'checkpoint': str(checkpoint_path)},
        )
        for policy_name, module in _load_policy_modules(checkpoint_path).items()
    ]


class PolicyExportCallback(Callback):
    """Exports the policies of every checkpoint Tune saves, with the environment configuration they were trained on.

    A failed export is logged and the training goes on: losing hours of training to an export would be worse than a
    checkpoint without its exported policy (scripts/export_policies.py exports it afterwards).
    """

    def __init__(self, environment_configuration: Any):
        self.environment_configuration = environment_configuration

    def on_checkpoint(self, iteration: int, trials: list, trial, checkpoint, **info) -> None:
        try:
            export_checkpoint_policies(
                Path(checkpoint.path),
                {
                    'environment_configuration': self.environment_configuration,
                    'training_iteration': (trial.last_result or {}).get('training_iteration'),
                },
            )
        except Exception:
            logger.exception('The policies of the checkpoint %s were not exported', checkpoint.path)
