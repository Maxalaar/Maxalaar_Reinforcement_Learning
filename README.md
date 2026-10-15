# Maxalaar Reinforcement Learning

A thin layer over [Ray RLlib](https://docs.ray.io/en/latest/rllib/): experiment configurations in Python,
training and resuming, watching and filming trained agents, exporting them (ONNX, ncnn), and a set of
environments. It is the reinforcement learning part of the thesis repository `Clustering_Agent_Latent_Space`,
moved out on its own; the thesis analyses (clustering, surrogate policies, latent space, semantic analysis)
stay there.

## Installation

```bash
conda create -n Maxalaar_Reinforcement_Learning python=3.13
conda activate Maxalaar_Reinforcement_Learning
pip install -e ".[development,export]"     # add environment extras as needed, see below
python scripts/install_ray_patch.py         # once per environment, see "Ray patch"
```

On NixOS, OpenCV (imported by Ray) needs `libgthread`, which pip packages cannot find on their own:
`conda install -n Maxalaar_Reinforcement_Learning glib` provides it.

Environments with extra dependencies are optional extras, installed only when needed:

| Extra | Environments |
|---|---|
| `box2d` | LunarLander, BipedalWalker, CarRacing |
| `mujoco` | Ant |
| `atari` | Atari |
| `flappy-bird`, `tetris`, `mini-grid`, `vizdoom` | the environment of the same name |
| `godot` | Godot environments |
| `rocket-league` | Rocket League (rlgym) |
| `all-environments` | all of the above |

Cubeball is a separate package: `pip install -e /media/SSD_1To/Programing_Projects/Python/Cubeball_Environment`.
Environments are registered by name in `environments/register_environments.py` and imported only when used,
so a missing optional dependency only affects its own environment.

## Usage

Scripts are run from the project root; results go to `experiments/<experimentation_name>/`.

```bash
python scripts/reinforcement_learning_training.py --experimentation_configuration_file configurations/cart_pole/cart_pole.py
python scripts/continue_reinforcement_learning_training.py \
    --experimentation_configuration_file configurations/cart_pole/cart_pole.py \
    --reinforcement_learning_path experiments/cart_pole/reinforcement_learning/base/version_0
python scripts/display_episodes.py --experimentation_configuration_file ... --reinforcement_learning_path ...
python scripts/video_episodes_generation.py --experimentation_configuration_file ... --reinforcement_learning_path ...
python scripts/export_reinforcement_learning_module.py --checkpoints_source experiments/cart_pole --export_formats onnx
python scripts/monitoring.py
python scripts/system_resource_monitoring.py
```

A configuration file defines an `ExperimentationConfiguration` in a variable named like the file
(`configurations/cart_pole/cart_pole.py` defines `cart_pole`). Weights & Biases logging needs
`./secrets/wandb_api_key.txt`.

## Layout

```
src/maxalaar_reinforcement_learning/
├── configuration/     ExperimentationConfiguration, ReinforcementLearningConfiguration, display and video settings
├── training/          building the RLlib algorithm configuration
├── architectures/     RLModules: dense, CNN, LSTM, transformer (PPO, APPO, DQN)
├── connectors/        connectors logging observations and actions
├── callbacks/         memory usage callback
├── checkpoints/       finding the best checkpoint, reading its configuration
├── export/            ONNX and ncnn export
├── patches/           Ray fixes (see below)
├── environments/      registry, wrappers, one folder per environment
├── neural_network/    building blocks and network visualizer
└── utilities/         Ray initialisation, versions of runs, saving
scripts/               the programs above
configurations/        experiments: cart_pole, cubeball
cluster/criann/        CRIANN submission scripts (not yet run from this project, see its README)
tests/                 end-to-end checks on CartPole (they start Ray and train, about a minute)
```

## Ray patch

Resuming a training from a checkpoint could crash with
`ValueError: beta1 as a Tensor is not supported for capturable=False and foreach=True`: RLlib restores Adam's
scalar hyperparameters as tensors. `patches/torch_learner_optimizer_state_restore.py` fixes the faulty method in
memory, in every process. `scripts/install_ray_patch.py` writes a `.pth` file in site-packages so that Python
installs the fix at startup, Ray workers included. It replaces the `sitecustomize.py` of the thesis repository:
conda's Python 3.13 ships its own `sitecustomize.py`, which hid that one. `tests/test_training.py` checks that a
restored training runs on the GPU. After a Ray upgrade, run the tests: if RLlib renames
`TorchLearner._set_optimizer_state`, the patch stops applying.

## Known issue

After a restore, an agent using `DensePPO` and evaluated on the local environment runner
(`evaluation_num_environment_runners = 0`) gets its inputs on the GPU and its weights on the CPU, and the
evaluation crashes. It already happens in the thesis repository with Ray 2.48. With at least one evaluation
runner it works. `tests/test_training.py` keeps an expected-failure test that will report when it is fixed.

## Development

```bash
pytest                     # tests
ruff check . && ruff format --check .
```
