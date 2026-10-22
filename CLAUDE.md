# Maxalaar_Reinforcement_Learning

Thin personal layer over Ray RLlib (configurations, training, evaluation, export, environments), extracted on
2026-10-15 from the thesis repository /media/SSD_1To/Programing_Projects/Python/Clustering_Agent_Latent_Space
(which stays untouched as the thesis archive; thesis-only code stayed there).

## Conventions
- Discuss with the user in French. Names in full, no acronyms in file, folder and identifier names.
- Conda environment `Maxalaar_Reinforcement_Learning` (Python 3.13, /media/SSD_1To/conda/envs/), `pip install -e .`;
  `glib` installed by conda for OpenCV on NixOS; conda variable `TORCH_DISABLE_NATIVE_JIT=1`
  (torch 2.14 routes some GPU ops, such as 1×1 attention, to Triton kernels that need /sbin/ldconfig and a C compiler,
  both absent on NixOS). Ray pinned to 2.54.0 in pyproject.toml.
- Scripts run from the project root (paths like `experiments/` are relative to the working directory).
- Configuration files define a variable named like the file (`get_configuration_class` relies on it).

## Decisions
- Environments are lazily imported through `environments/register_environments.py` (name -> "module:attribute");
  optional dependencies are pyproject extras.
- The Ray optimizer-restore patch is installed by a .pth file (`scripts/install_ray_patch.py`), not by
  `ray.init(runtime_env=...)`: the user chose to avoid per-actor setup cost. sitecustomize.py does not work with
  conda's Python 3.13, which ships one.
- `ExperimentationConfiguration` only holds reinforcement learning fields; other projects subclass it.
- Next planned user: the Beyond All Reason environment
  (/media/SSD_1To/Programing_Projects/Python/Beyond_All_Reason_Reinforcement_Learning), as a MultiAgentEnv adapter here.

## Known issues / follow-ups
- DensePPO + restore + evaluation on the local env runner: device mismatch (pre-existing with Ray 2.48); xfail test.
- Machine profiles (resources separate from experiments) not done yet; only cart_pole and cubeball configurations
  were migrated; cluster/criann scripts updated but not run.
- Lint is limited to real errors (pyproject [tool.ruff.lint]) because the code comes as is from the thesis.
- environments/slime_volley/play.py uses LatentSpace2DVisualizer, defined nowhere (already broken in the thesis).
