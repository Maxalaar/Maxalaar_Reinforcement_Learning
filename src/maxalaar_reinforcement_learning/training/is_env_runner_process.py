import ray


def is_env_runner_process() -> bool:
    # `RLModule.inference_only` is unreliable here: RLlib silently drops it when a custom
    # per-policy RLModuleSpec is provided (e.g. via `initialization_checkpoint_path`,
    # which sets `load_state_path`), leaving it False on EnvRunners too. Ray's own actor
    # task name, unlike RLlib's flag, always correctly reflects which actor is running.
    runtime_context = ray.get_runtime_context()
    get_task_function_name = getattr(runtime_context, 'get_task_function_name', None)
    if get_task_function_name is None:
        # Older Ray releases (e.g. 2.39, the version the Atari checkpoints were trained
        # with) don't expose `get_task_function_name`. Fall back to "not an EnvRunner"
        # so callers still emit their diagnostics (pre-existing behaviour).
        return False
    task_function_name = get_task_function_name() or ''
    return 'EnvRunner' in task_function_name
