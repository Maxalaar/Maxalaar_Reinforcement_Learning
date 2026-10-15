import os
from pathlib import Path
from typing import Dict, List, Optional

import yaml
from ray.rllib.core import DEFAULT_MODULE_ID
from ray.rllib.core.rl_module import RLModule, MultiRLModule

from maxalaar_reinforcement_learning.export.checkpoint_selection import select_checkpoints
from maxalaar_reinforcement_learning.export.ncnn_export import rl_module_to_ncnn
from maxalaar_reinforcement_learning.export.onnx_export import rl_module_to_onnx

RL_MODULE_EXPORTERS = {
    'onnx': rl_module_to_onnx,
    'ncnn': rl_module_to_ncnn,
}
EXPORT_FORMATS = tuple(RL_MODULE_EXPORTERS)

# Mirrors `ExperimentationConfiguration`.
REINFORCEMENT_LEARNING_DIRECTORY_NAME = 'reinforcement_learning'
EXPORT_DIRECTORY_NAME = 'export'

# The learner's module, restored into the env runners on `Algorithm.restore()`.
RL_MODULE_CHECKPOINT_SUBPATH = Path('learner_group') / 'learner' / 'rl_module'


def _load_policy_modules(checkpoint_path: Path) -> Dict[str, RLModule]:
    """Only the module: no environment (it would launch the game), no Ray cluster."""
    module_checkpoint_path = Path(checkpoint_path).resolve() / RL_MODULE_CHECKPOINT_SUBPATH
    if not module_checkpoint_path.is_dir():
        raise ValueError('No module checkpoint in ' + str(module_checkpoint_path))
    module = RLModule.from_checkpoint(str(module_checkpoint_path))

    if issubclass(type(module), MultiRLModule):
        policy_modules = {policy_name: module[policy_name] for policy_name in module.keys()}
    else:
        policy_modules = {DEFAULT_MODULE_ID: module}
    return {policy_name: policy_module.cpu().eval() for policy_name, policy_module in policy_modules.items()}


def _export_checkpoint(
    checkpoint_path: Path,
    export_paths: Dict[str, Path],
    version_name: Optional[str],
) -> Dict[str, Dict[str, List[Path]]]:
    """Flat `<dir>/<policy>.*` without `version_name`, else `<dir>/<policy>/<policy>_<version>.*`.
    Returns policy -> format -> written files."""
    exported_files: Dict[str, Dict[str, List[Path]]] = {}
    for policy_name, policy_module in _load_policy_modules(checkpoint_path).items():
        exported_files[policy_name] = {}
        for export_format, export_path in export_paths.items():
            policy_export_path = export_path if version_name is None else export_path / policy_name
            os.makedirs(policy_export_path, exist_ok=True)
            exported_files[policy_name][export_format] = RL_MODULE_EXPORTERS[export_format](
                policy_module,
                policy_name,
                policy_export_path,
                file_stem=None if version_name is None else policy_name + '_' + version_name,
            )
    return exported_files


def _experiment_root(checkpoint_path: Path) -> Optional[Path]:
    for parent in checkpoint_path.parents:
        if parent.name == REINFORCEMENT_LEARNING_DIRECTORY_NAME:
            return parent.parent
    return None


def _default_export_paths(checkpoint_paths: List[Path], export_formats: List[str]) -> Dict[str, Path]:
    """`<experiment>/export/<format>/`, whatever the source given inside the experiment."""
    experiment_roots = {_experiment_root(checkpoint_path) for checkpoint_path in checkpoint_paths}
    if None in experiment_roots or len(experiment_roots) != 1:
        raise ValueError(
            'Cannot deduce the export directory, the checkpoints are not below a single '
            '<experiment>/' + REINFORCEMENT_LEARNING_DIRECTORY_NAME + '/ directory '
            '(found: ' + ', '.join(sorted(str(root) for root in experiment_roots)) + '). '
            'Give an output directory.'
        )
    experiment_root = experiment_roots.pop()
    return {export_format: experiment_root / EXPORT_DIRECTORY_NAME / export_format for export_format in export_formats}


def _write_manifests(export_paths: Dict[str, Path], manifest_entries: List[dict], header: dict) -> None:
    """One `manifest.yaml` per export directory."""
    for directory in sorted(set(export_paths.values())):
        directory_formats = [export_format for export_format, path in export_paths.items() if path == directory]
        entries = []
        for entry in manifest_entries:
            entries.append({
                **{key: value for key, value in entry.items() if key != 'files'},
                'files': {
                    policy_name: {
                        export_format: [str(path.relative_to(directory)) for path in format_files[export_format]]
                        for export_format in directory_formats
                    }
                    for policy_name, format_files in entry['files'].items()
                },
            })

        manifest_path = directory / 'manifest.yaml'
        with open(manifest_path, 'w') as manifest_file:
            yaml.dump(
                {**header, 'export_formats': directory_formats, 'checkpoints': entries},
                manifest_file,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
            )
        print('Wrote manifest: ' + str(manifest_path))


def export_rl_module(
    checkpoints_sources: List[Path],
    output_directory: Optional[Path] = None,
    selection: str = 'best',
    order_by: str = 'creation_time',
    export_formats: List[str] = EXPORT_FORMATS,
):
    print('Ordering checkpoints by "' + order_by + '"')
    print()

    selected_checkpoints, found_checkpoint_paths = select_checkpoints(checkpoints_sources, selection, order_by)

    print()
    print('Found ' + str(len(found_checkpoint_paths)) + ' checkpoints, '
          + str(len(selected_checkpoints)) + ' to export (selection: ' + selection + ').')
    print()

    if output_directory is None:
        export_paths = _default_export_paths(found_checkpoint_paths, export_formats)
    else:
        export_paths = {export_format: output_directory for export_format in export_formats}
    for export_path in export_paths.values():
        os.makedirs(export_path, exist_ok=True)

    manifest_entries = []
    for selected in selected_checkpoints:
        checkpoint = selected.checkpoint
        print('=== ' + selected.version_name
              + ' (' + selected.score_criterion.metric + ' = ' + str(checkpoint.metric_value)
              + ', date = ' + str(checkpoint.creation_time) + ') ===')
        print('Checkpoint: ' + str(checkpoint.path))

        exported_files = _export_checkpoint(
            checkpoint_path=checkpoint.path,
            export_paths=export_paths,
            version_name=None if selection == 'best' else selected.version_name,
        )

        manifest_entries.append({
            'version': selected.version_name,
            'checkpoint_path': str(checkpoint.path),
            'metric': selected.score_criterion.metric,
            'metric_mode': selected.score_criterion.mode,
            'metric_value': checkpoint.metric_value,
            'creation_time': None if checkpoint.creation_time is None else checkpoint.creation_time.isoformat(),
            'files': exported_files,
        })

    print()
    _write_manifests(
        export_paths,
        manifest_entries,
        header={
            'selection': selection,
            'order_by': order_by,
            'number_checkpoints_found': len(found_checkpoint_paths),
        },
    )
