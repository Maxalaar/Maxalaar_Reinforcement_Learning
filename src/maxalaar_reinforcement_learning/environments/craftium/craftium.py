import os
import gymnasium as gym
from gymnasium.wrappers import TransformReward

from maxalaar_reinforcement_learning.environments.wrappers.apply_rgb_wrappers import apply_rgb_wrappers
from maxalaar_reinforcement_learning.environments.wrappers.resize_observation import ResizeObservation

# https://github.com/mikelma/craftium/blob/main/craftium/craftium_env.py
def make_craftium(environment_configuration):
    import craftium

    environment_name = environment_configuration.get('environment_name')
    environment_id = 'Craftium/'+environment_name
    rgb_observations = environment_configuration.get('rgb_observations', False)
    frameskip = environment_configuration.get('frameskip', 4)
    observation_width = environment_configuration.get('observation_width', 64)
    observation_height = environment_configuration.get('observation_height', 64)
    human_screen_size = environment_configuration.get('human_screen_size', (720, 720))
    render_mode = environment_configuration.get('render_mode', None)
    reward_scale_factor = environment_configuration.get('reward_scale_factor', 1.0)
    max_timesteps = environment_configuration.get('max_timesteps', None)
    minetest_configuration = environment_configuration.get('minetest_configuration', {})

    run_dir_prefix = './temporary/craftium'
    os.makedirs(run_dir_prefix, exist_ok=True)

    environment = gym.make(
        environment_id,
        run_dir_prefix = run_dir_prefix,
        render_mode = render_mode,
        rgb_observations = rgb_observations,
        frameskip = frameskip,
        obs_width = observation_height,
        obs_height = observation_width,
        max_timesteps=max_timesteps,
        human_screen_size = human_screen_size,
        minetest_conf=minetest_configuration,
    )
    # environment = ResizeObservation(environment, (64, 64))
    environment = apply_rgb_wrappers(environment, environment_configuration)
    environment = TransformReward(environment, lambda r: reward_scale_factor * r)

    return environment

