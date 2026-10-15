import sys
import importlib.abc
import importlib.util

_TARGET_MODULE_NAME = 'ray.rllib.core.learner.torch.torch_learner'
_PATCH_MARKER = '_optimizer_scalar_restore_patched'


def _sanitize_optimizer_param_groups(optimizer) -> None:
    import torch

    for group in optimizer.param_groups:
        for key in ('lr', 'eps', 'weight_decay'):
            value = group.get(key)
            if isinstance(value, torch.Tensor):
                group[key] = value.item()
        betas = group.get('betas')
        if betas is not None and any(isinstance(beta, torch.Tensor) for beta in betas):
            group['betas'] = tuple(
                beta.item() if isinstance(beta, torch.Tensor) else beta
                for beta in betas
            )


def _patch_torch_learner_module(module) -> None:
    learner_class = module.TorchLearner
    if getattr(learner_class, _PATCH_MARKER, False):
        return

    original_set_optimizer_state = learner_class._set_optimizer_state

    def _patched_set_optimizer_state(self, state):
        original_set_optimizer_state(self, state)
        for name in state:
            optimizer = self._named_optimizers.get(name)
            if optimizer is not None:
                _sanitize_optimizer_param_groups(optimizer)

    learner_class._set_optimizer_state = _patched_set_optimizer_state
    setattr(learner_class, _PATCH_MARKER, True)


class _PatchedLoader(importlib.abc.Loader):
    def __init__(self, wrapped_loader):
        self._wrapped_loader = wrapped_loader

    def create_module(self, spec):
        return self._wrapped_loader.create_module(spec)

    def exec_module(self, module) -> None:
        self._wrapped_loader.exec_module(module)
        _patch_torch_learner_module(module)


class _PatchFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname != _TARGET_MODULE_NAME:
            return None
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(fullname)
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return None
        spec.loader = _PatchedLoader(spec.loader)
        return spec


def install() -> None:
    """
    Work around a Ray RLlib bug where restoring a checkpointed optimizer's
    `state_dict` (via `convert_to_torch_tensor`) turns scalar hyperparameters
    such as Adam's `betas`/`lr`/`eps` into 0-dim tensors. Recent torch
    versions reject a Tensor `beta1` in the foreach/multi-tensor Adam path on
    GPU: "beta1 as a Tensor is not supported for capturable=False and
    foreach=True".

    This installs a `sys.meta_path` import hook that fixes up
    `TorchLearner._set_optimizer_state` the moment RLlib's module is
    imported, in any process (driver or Ray worker). Unlike editing Ray's
    installed source directly, this survives a `pip install --upgrade ray`.
    Unlike routing the patch through `ray.init(runtime_env=...)`, it doesn't
    go through Ray's runtime-env setup machinery for every actor, so it adds
    no measurable per-actor overhead.
    """
    if any(isinstance(finder, _PatchFinder) for finder in sys.meta_path):
        return
    if _TARGET_MODULE_NAME in sys.modules:
        _patch_torch_learner_module(sys.modules[_TARGET_MODULE_NAME])
        return
    sys.meta_path.insert(0, _PatchFinder())
