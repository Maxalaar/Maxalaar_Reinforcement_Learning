"""Export the policies of existing checkpoints, as a training does at each checkpoint with
export_policies_at_checkpoint: <checkpoint>/policies/<policy>/policy.onnx and policy.json (export/policy_export.py).

To do while the code the checkpoints were made with still loads them. Every checkpoint below the given paths (a
folder with rllib_checkpoint.json and learner_group/) is exported, an already exported one again.
"""

import argparse
from pathlib import Path

from maxalaar_reinforcement_learning.checkpoints.get_checkpoint_algorithm_configuration import (
    get_checkpoint_algorithm_configuration,
)
from maxalaar_reinforcement_learning.export.policy_export import export_checkpoint_policies


def checkpoints_below(path: Path) -> list[Path]:
    found = path.rglob('rllib_checkpoint.json') if path.is_dir() else []
    return sorted(
        file.parent for file in [*found, path / 'rllib_checkpoint.json']
        if file.is_file() and (file.parent / 'learner_group').is_dir()
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('paths', type=Path, nargs='+', help='Checkpoints, or folders searched for checkpoints.')
    arguments = parser.parse_args()

    checkpoints = sorted({checkpoint for path in arguments.paths for checkpoint in checkpoints_below(path.resolve())})
    if not checkpoints:
        raise SystemExit('No checkpoint below ' + ', '.join(str(path) for path in arguments.paths))
    for checkpoint in checkpoints:
        environment_configuration = get_checkpoint_algorithm_configuration(checkpoint).env_config
        for directory in export_checkpoint_policies(checkpoint, {'environment_configuration': environment_configuration}):
            print('Exported ' + str(directory))
