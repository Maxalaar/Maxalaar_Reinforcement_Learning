"""Policies exported to ONNX with their description (export/policy_export.py), without Ray nor training."""

import dataclasses
import enum
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch

pytest.importorskip('onnxruntime')
pytest.importorskip('onnxscript')

import onnxruntime  # noqa: E402
from ray.rllib.core.columns import Columns  # noqa: E402
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec  # noqa: E402
from ray.rllib.core.rl_module.rl_module import RLModuleSpec  # noqa: E402
from ray.rllib.core.rl_module.torch import TorchRLModule  # noqa: E402

from maxalaar_reinforcement_learning.export.policy_export import (  # noqa: E402
    DESCRIPTION_FILE_NAME,
    ONNX_FILE_NAME,
    OUTPUT_NAME,
    POLICIES_DIRECTORY_NAME,
    export_checkpoint_policies,
    export_policy,
    sample_observations,
    to_json,
)



@pytest.fixture(autouse=True)
def no_ray_left_running():
    """After the trainings of other test files, an RLlib call of the export can start Ray: stop it, ray.init of the
    next tests would fail."""
    import ray

    running_before = ray.is_initialized()
    yield
    if ray.is_initialized() and not running_before:
        ray.shutdown()


OBSERVATION_SPACE = gym.spaces.Dict(
    {
        'units': gym.spaces.Dict({'position': gym.spaces.Box(-1, 1, (3, 2), np.float32)}),
        'command_mask': gym.spaces.MultiBinary(3),
        'health': gym.spaces.Box(0, 1, (1,), np.float32),
    }
)


class MaskedPolicy(TorchRLModule):
    """Logits of 3 actions from a nested observation, the actions forbidden by command_mask at −∞."""

    def setup(self):
        self.layer = torch.nn.Linear(3 * 2 + 1, 3)

    def _forward(self, batch, **kwargs):
        observation = batch[Columns.OBS]
        features = torch.cat([observation['units']['position'].flatten(1), observation['health']], dim=-1)
        logits = self.layer(features).masked_fill(observation['command_mask'] == 0, float('-inf'))
        return {Columns.ACTION_DIST_INPUTS: logits}


def policy() -> MaskedPolicy:
    torch.manual_seed(0)
    return MaskedPolicy(observation_space=OBSERVATION_SPACE, action_space=gym.spaces.Discrete(3), model_config={})


class Side(enum.Enum):
    west = 'west'


@dataclasses.dataclass
class Scenario:
    name: str
    side: Side
    controllers: list


def test_the_exported_policy_plays_as_the_module_without_its_code(tmp_path: Path):
    module = policy()
    export_policy(module, tmp_path, {'environment_configuration': {'observation_version': 1}})

    session = onnxruntime.InferenceSession(str(tmp_path / ONNX_FILE_NAME))
    observations = sample_observations(OBSERVATION_SPACE, batch_size=50, seed=7)
    obtained = session.run([OUTPUT_NAME], observations)[0]
    with torch.no_grad():
        expected = module.forward_inference(
            {
                Columns.OBS: {
                    'units': {'position': torch.from_numpy(observations['units.position'])},
                    'command_mask': torch.from_numpy(observations['command_mask']),
                    'health': torch.from_numpy(observations['health']),
                }
            }
        )[Columns.ACTION_DIST_INPUTS].numpy()

    assert obtained.shape == (50, 3)
    assert np.array_equal(np.isinf(obtained), observations['command_mask'] == 0)
    assert np.allclose(obtained[np.isfinite(expected)], expected[np.isfinite(expected)], atol=1e-5)


def test_the_description_says_what_the_policy_expects(tmp_path: Path):
    export_policy(policy(), tmp_path, {'environment_configuration': {'observation_version': 1}})
    description = json.loads((tmp_path / DESCRIPTION_FILE_NAME).read_text())

    assert [(entry['name'], entry['shape'], entry['dtype']) for entry in description['inputs']] == [
        ('command_mask', [3], 'int8'),
        ('health', [1], 'float32'),
        ('units.position', [3, 2], 'float32'),
    ]
    assert description['output'] == {'name': OUTPUT_NAME, 'size': 3}
    assert description['environment_configuration'] == {'observation_version': 1}
    assert description['module_class'].endswith('test_policy_export.MaskedPolicy')
    assert description['action_space']['space'] == 'discrete'
    commit = next(iter(description['source_commits'].values()))
    assert len(commit['commit']) == 40


def test_every_policy_of_a_checkpoint_is_exported_beside_it(tmp_path: Path):
    specification = RLModuleSpec(
        module_class=MaskedPolicy, observation_space=OBSERVATION_SPACE, action_space=gym.spaces.Discrete(3)
    )
    MultiRLModuleSpec(rl_module_specs={'shared_policy': specification}).build().save_to_path(
        tmp_path / 'learner_group' / 'learner' / 'rl_module'
    )

    directories = export_checkpoint_policies(tmp_path, {'training_iteration': 50})

    assert directories == [tmp_path.resolve() / POLICIES_DIRECTORY_NAME / 'shared_policy']
    description = json.loads((directories[0] / DESCRIPTION_FILE_NAME).read_text())
    assert (description['policy'], description['training_iteration']) == ('shared_policy', 50)
    assert (directories[0] / ONNX_FILE_NAME).is_file()


def test_a_configuration_is_written_as_json_data():
    configuration = {'scenario': Scenario('arena', Side.west, [Scenario, Path('a/b')]), 'size': np.float32(2.5)}
    assert to_json(configuration) == {
        'scenario': {
            'class': f'{__name__}.Scenario',
            'name': 'arena',
            'side': 'west',
            'controllers': [f'{__name__}.Scenario', 'a/b'],
        },
        'size': 2.5,
    }


def test_a_recurrent_policy_is_refused(tmp_path: Path, monkeypatch):
    module = policy()
    monkeypatch.setattr(module, 'is_stateful', lambda: True)
    with pytest.raises(ValueError, match='recurrent'):
        export_policy(module, tmp_path)
