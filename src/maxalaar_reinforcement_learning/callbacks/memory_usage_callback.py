import gc
import os
from collections import Counter
from typing import Type

import psutil
import torch
from ray.rllib.algorithms.callbacks import DefaultCallbacks

# Module-level, so it persists across calls on the same long-lived env runner actor
# process (each `foreach_env_runner` call below runs this file's code inside that
# actor, reusing whatever this module already has in memory from the previous call).
_previous_object_type_counts = None


def _process_rss_megabytes() -> float:
    return psutil.Process().memory_info().rss / (1024 ** 2)


def _children_rss_megabytes() -> float:
    total = 0
    for child in psutil.Process().children(recursive=True):
        try:
            total += child.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return total / (1024 ** 2)


def _object_type_growth_since_last_call(top_n: int = 8) -> dict:
    # Diffs the live object-type census against the previous call on this same process,
    # so a Python-side leak shows up as a type whose count only ever goes up. gc.collect()
    # first so we're not counting garbage that a collection would have reclaimed anyway.
    # Expensive (forces a full GC pass and walks every live object every call) -- opt-in
    # only, see `log_env_runner_object_type_growth` on `create_memory_usage_callback`.
    global _previous_object_type_counts
    gc.collect()
    current_counts = Counter(type(obj).__name__ for obj in gc.get_objects())

    if _previous_object_type_counts is None:
        _previous_object_type_counts = current_counts
        return {}

    deltas = {
        name: current_counts[name] - _previous_object_type_counts.get(name, 0)
        for name in current_counts
    }
    _previous_object_type_counts = current_counts

    growing = sorted(
        ((name, delta) for name, delta in deltas.items() if delta > 0),
        key=lambda item: item[1],
        reverse=True,
    )
    return dict(growing[:top_n])


def _env_runner_memory_snapshot(log_object_type_growth: bool) -> dict:
    # `children` is the env's own subprocess (e.g. the Godot game binary launched by
    # CubeballConnection) -- separate from `own` (the env runner's Python process itself)
    # so a leak inside the game engine can be told apart from one in the Python glue code.
    if log_object_type_growth:
        growing_types = _object_type_growth_since_last_call()
        if growing_types:
            print(f"MEMDIAG env_runner pid={os.getpid()} growing_object_types={growing_types}", flush=True)

    return {
        'own': _process_rss_megabytes(),
        'children': _children_rss_megabytes(),
    }


def _learner_memory_snapshot() -> dict:
    snapshot = {'own': _process_rss_megabytes()}

    if torch.cuda.is_available():
        # `reserved` (the caching allocator's pool) growing while `allocated` (actually
        # live tensors) stays flat is normal PyTorch behavior, not a leak -- comparing the
        # two tells them apart.
        snapshot['cuda_allocated'] = torch.cuda.memory_allocated() / (1024 ** 2)
        snapshot['cuda_reserved'] = torch.cuda.memory_reserved() / (1024 ** 2)

    # Coarse but direct: a live tensor count that keeps climbing points straight at a
    # reference leak (e.g. a tensor kept `requires_grad=True` and never released).
    snapshot['live_tensor_count'] = sum(1 for obj in gc.get_objects() if torch.is_tensor(obj))

    return snapshot


class MemoryUsageCallback(DefaultCallbacks):
    # Kept light by default (no forced `gc.collect()` / object census) so it's cheap
    # enough to leave on for every run -- see `create_memory_usage_callback` to opt into
    # the more expensive diagnostics only while actively hunting a leak.
    _log_env_runner_object_type_growth = False

    # Also logged on init (not just on_train_result) so the `memory_usage_mb/*` keys
    # are already present in the very first training result. Ray Tune's CSVLoggerCallback
    # fixes progress.csv's columns from the first row it writes, so a key that only
    # starts appearing later would silently be dropped from progress.csv (TensorBoard
    # is unaffected either way, since it logs scalars per-call rather than per-row).
    def on_algorithm_init(self, *, algorithm, metrics_logger=None, **kwargs):
        self._log_memory_usage(algorithm, metrics_logger)

    def on_train_result(self, *, algorithm, metrics_logger=None, result, **kwargs):
        self._log_memory_usage(algorithm, metrics_logger)

    def _log_memory_usage(self, algorithm, metrics_logger):
        metrics_logger.log_value(('memory_usage_mb', 'driver'), _process_rss_megabytes(), reduce='mean', window=1)

        log_object_type_growth = self._log_env_runner_object_type_growth
        env_runners_snapshots = algorithm.env_runner_group.foreach_env_runner(
            lambda env_runner: _env_runner_memory_snapshot(log_object_type_growth),
            local_env_runner=False,
        )
        if env_runners_snapshots:
            own_values = [snapshot['own'] for snapshot in env_runners_snapshots]
            children_values = [snapshot['children'] for snapshot in env_runners_snapshots]
            metrics_logger.log_value(('memory_usage_mb', 'env_runners_mean'), sum(own_values) / len(own_values), reduce='mean', window=1)
            metrics_logger.log_value(('memory_usage_mb', 'env_runners_max'), max(own_values), reduce='mean', window=1)
            metrics_logger.log_value(('memory_usage_mb', 'env_runners_godot_mean'), sum(children_values) / len(children_values), reduce='mean', window=1)
            metrics_logger.log_value(('memory_usage_mb', 'env_runners_godot_max'), max(children_values), reduce='mean', window=1)

        learners_snapshots = [
            call_result.get()
            for call_result in algorithm.learner_group.foreach_learner(lambda learner: _learner_memory_snapshot()).ignore_errors()
        ]
        if learners_snapshots:
            own_values = [snapshot['own'] for snapshot in learners_snapshots]
            metrics_logger.log_value(('memory_usage_mb', 'learners_mean'), sum(own_values) / len(own_values), reduce='mean', window=1)
            metrics_logger.log_value(('memory_usage_mb', 'learners_max'), max(own_values), reduce='mean', window=1)

            if 'cuda_allocated' in learners_snapshots[0]:
                allocated_values = [snapshot['cuda_allocated'] for snapshot in learners_snapshots]
                reserved_values = [snapshot['cuda_reserved'] for snapshot in learners_snapshots]
                metrics_logger.log_value(('memory_usage_mb', 'learners_cuda_allocated_mean'), sum(allocated_values) / len(allocated_values), reduce='mean', window=1)
                metrics_logger.log_value(('memory_usage_mb', 'learners_cuda_reserved_mean'), sum(reserved_values) / len(reserved_values), reduce='mean', window=1)

            tensor_counts = [snapshot['live_tensor_count'] for snapshot in learners_snapshots]
            metrics_logger.log_value(('memory_usage_diagnostics', 'learners_live_tensor_count_mean'), sum(tensor_counts) / len(tensor_counts), reduce='mean', window=1)


def create_memory_usage_callback(
        log_env_runner_object_type_growth: bool = False,
) -> Type[MemoryUsageCallback]:
    # Factory so the (expensive) object-census diagnostic can be toggled per-experiment
    # without touching this file -- e.g.
    # `reinforcement_learning_configuration.callback = create_memory_usage_callback(log_env_runner_object_type_growth=True)`
    # RLlib instantiates whatever `.callbacks(...)` is given with zero arguments, so the
    # option has to live on the class itself rather than be passed to `__init__`.
    class ConfiguredMemoryUsageCallback(MemoryUsageCallback):
        _log_env_runner_object_type_growth = log_env_runner_object_type_growth

    return ConfiguredMemoryUsageCallback
