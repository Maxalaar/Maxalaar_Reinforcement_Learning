import gymnasium

from ray.tune.registry import _Registry, register_env


def register_human_rendering_environment_creator(environment_name: str):
    human_rendering_environment_name = environment_name + 'HumanRendering'

    def human_rendering_environment_creator(configuration: dict):
        environment_creator = _Registry().get('env_creator', environment_name)
        environment = environment_creator(configuration)
        if environment.render_mode is not None and environment.render_mode != 'human':
            human_rendering_environment = gymnasium.wrappers.HumanRendering(environment)
        else:
            human_rendering_environment = environment
        return human_rendering_environment

    register_env(name=human_rendering_environment_name, env_creator=human_rendering_environment_creator)

    return human_rendering_environment_name
