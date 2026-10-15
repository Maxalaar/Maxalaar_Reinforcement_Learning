"""Imported at the start of every Python process by the .pth file of scripts/install_ray_patch.py.

Installs the Ray fixes of this package before RLlib is imported, in the driver and in every Ray worker.
Never raises: a failure only prints a warning, so it cannot break unrelated Python programs.
"""

import sys

try:
    from maxalaar_reinforcement_learning.patches.torch_learner_optimizer_state_restore import install

    install()
except Exception as error:
    sys.stderr.write(f"[maxalaar_reinforcement_learning] Could not install the Ray optimizer-restore patch: {error!r}\n")
