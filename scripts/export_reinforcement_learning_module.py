import argparse
from pathlib import Path

from maxalaar_reinforcement_learning.export.checkpoint_selection import ORDER_BY_CHOICES, SELECTION_CHOICES
from maxalaar_reinforcement_learning.export.export_reinforcement_learning_module import EXPORT_FORMATS, export_rl_module


def _absolute(path: str) -> Path:
    path = Path(path)
    return path if path.is_absolute() else Path.cwd() / path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export the RL modules of training checkpoints to ONNX and/or ncnn.')
    parser.add_argument('--checkpoints_source', type=str, nargs='+')
    parser.add_argument(
        '--output_directory',
        type=str,
        default=None,
        help='Default: <experiment>/export/<format>.',
    )
    parser.add_argument(
        '--selection',
        type=str,
        choices=SELECTION_CHOICES,
        default='best',
        help='best (default): only the strongest agent, written flat. all: every checkpoint, versioned.',
    )
    parser.add_argument(
        '--order_by',
        type=str,
        choices=ORDER_BY_CHOICES,
        default='creation_time',
        help='creation_time (default): robust to chained runs. metric: the checkpoint score criterion of the runs.',
    )
    parser.add_argument('--export_formats', type=str, nargs='+', choices=EXPORT_FORMATS, default=list(EXPORT_FORMATS))

    arguments = parser.parse_args()

    export_rl_module(
        checkpoints_sources=[_absolute(source) for source in arguments.checkpoints_source],
        output_directory=None if arguments.output_directory is None else _absolute(arguments.output_directory),
        selection=arguments.selection,
        order_by=arguments.order_by,
        export_formats=arguments.export_formats,
    )
