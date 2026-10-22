"""Environment names usable in an ExperimentationConfiguration, registered in Ray.

Each name points to "module:attribute" and the module is only imported when an environment is created,
so an environment whose optional dependencies are not installed (VizDoom, Rocket League, Godot...) does not
prevent the others from working. Install them with the extras of pyproject.toml, for example
`pip install -e ".[vizdoom]"`.

An environment of another package needs no entry: its class path ("package.module.Class") is a valid name, registered
by register_environment when an algorithm is built (build_algorithm_configuration) or a checkpoint is watched.
"""

import importlib
from typing import Callable

from ray.tune.registry import register_env

ENVIRONMENTS: dict[str, str] = {
    'CartPoleRllib': 'maxalaar_reinforcement_learning.environments.cart_pole.cart_pole:CartPole',
    'LunarLanderRllib': 'maxalaar_reinforcement_learning.environments.lunar_lander.lunar_lander:LunarLander',
    'FlappyBirdRllib': 'maxalaar_reinforcement_learning.environments.flappy_bird.flappy_bird:FlappyBird',
    'BipedalWalkerRllib': 'maxalaar_reinforcement_learning.environments.bipedal_walker.bipedal_walker:BipedalWalker',
    'AntRllib': 'maxalaar_reinforcement_learning.environments.ant.ant:Ant',
    'TaxiRllib': 'maxalaar_reinforcement_learning.environments.taxi.taxi:Taxi',
    'PongSurvivor': 'maxalaar_reinforcement_learning.environments.pong_survivor.pong_survivor:PongSurvivor',
    'TetrisRllib': 'maxalaar_reinforcement_learning.environments.tetris.tetris:Tetris',
    'AtariRllib': 'maxalaar_reinforcement_learning.environments.atari.atari:Atari',
    'SlimeVolleyRllib': 'maxalaar_reinforcement_learning.environments.slime_volley.slime_volley:SlimeVolley',
    'CarRacingRllib': 'maxalaar_reinforcement_learning.environments.car_racing.car_racing:CarRacing',
    'MiniGridRllib': 'maxalaar_reinforcement_learning.environments.mini_grid.mini_grid:MiniGrid',
    'CraftiumRllib': 'maxalaar_reinforcement_learning.environments.craftium.craftium:make_craftium',
    'ViZDoomRllib': 'maxalaar_reinforcement_learning.environments.vizdoom.vizdoom:ViZDoom',
    'GogotRllib': 'maxalaar_reinforcement_learning.environments.godot.godot:Gogot',
    'RocketLeague': 'maxalaar_reinforcement_learning.environments.rocket_league.rocket_league:RocketLeague',
    # Separate packages, installed on their own.
    'Cubeball': 'cubeball:Cubeball',
}


def load_environment_creator(name: str) -> Callable:
    """The class or function that creates the environment named ``name``: an entry of ENVIRONMENTS, or a class path."""
    if name in ENVIRONMENTS:
        module_name, attribute = ENVIRONMENTS[name].split(':')
    else:
        module_name, _, attribute = name.rpartition('.')
        if not module_name:
            raise ValueError(f"Unknown environment {name!r}: neither an entry of ENVIRONMENTS nor a class path.")
    return getattr(importlib.import_module(module_name), attribute)


def _lazy_creator(name: str) -> Callable:
    def create(environment_configuration):
        return load_environment_creator(name)(environment_configuration)

    return create


def register_environment(name: str):
    """Register ``name`` (an entry of ENVIRONMENTS or a class path) in Ray; its module is imported lazily."""
    register_env(name=name, env_creator=_lazy_creator(name))


def register_environments():
    for name in ENVIRONMENTS:
        register_environment(name)
