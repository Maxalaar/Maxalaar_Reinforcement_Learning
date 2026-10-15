"""Install the Ray RLlib optimizer-restore patch in the current Python environment (once per environment).

Resuming a PPO training from a checkpoint can crash with
"ValueError: beta1 as a Tensor is not supported for capturable=False and foreach=True": restoring the optimizer
state turns Adam's scalar hyperparameters into 0-dim tensors. The fix
(maxalaar_reinforcement_learning/patches/torch_learner_optimizer_state_restore.py) must run in every process,
Ray workers included, before RLlib's TorchLearner is imported.

Python executes the "import" lines of every .pth file in site-packages when it starts, so this script writes
one that imports maxalaar_reinforcement_learning.patches.startup. A .pth file is used rather than
sitecustomize.py because conda's Python 3.13 ships its own sitecustomize.py, which would hide ours. It
deliberately does not go through ray.init(runtime_env=...), which would add per-actor setup work.

Example:
    python scripts/install_ray_patch.py
"""

import sysconfig
from pathlib import Path

FILE_NAME = "maxalaar_reinforcement_learning_ray_patch.pth"
CONTENT = "import maxalaar_reinforcement_learning.patches.startup\n"


def main() -> None:
    path = Path(sysconfig.get_paths()["purelib"]) / FILE_NAME
    path.write_text(CONTENT)
    print(f"Installed: {path}")


if __name__ == "__main__":
    main()
