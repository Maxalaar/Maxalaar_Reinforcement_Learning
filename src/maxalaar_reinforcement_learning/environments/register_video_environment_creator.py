import string
import random
from pathlib import Path

import gymnasium

from ray.tune.registry import _Registry, register_env


class RandomString:
    def __init__(self, length=10):
        self.length = length

    def __str__(self):
        characters = string.ascii_letters + string.digits
        return ''.join(random.choice(characters) for _ in range(self.length))


def register_video_environment_creator(environment_name: str, video_episodes_storage_path: Path):
    video_environment_name = environment_name + 'Video'

    def video_environment_creator(configuration: dict):
        environment_creator = _Registry().get('env_creator', environment_name)
        random_string_instance = RandomString(12)
        environment = environment_creator(configuration)
        video_environment = gymnasium.wrappers.RecordVideo(
            env=environment,
            video_folder=str(video_episodes_storage_path),
            name_prefix=random_string_instance,
            episode_trigger=lambda x: True,
            disable_logger=True,
        )
        return video_environment

    register_env(name=video_environment_name, env_creator=video_environment_creator)

    return video_environment_name
