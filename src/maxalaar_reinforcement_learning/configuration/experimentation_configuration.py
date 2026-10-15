from pathlib import Path
from typing import Optional

import yaml

from maxalaar_reinforcement_learning.configuration.display_episodes_configuration import DisplayEpisodesConfiguration
from maxalaar_reinforcement_learning.configuration.reinforcement_learning_configuration import ReinforcementLearningConfiguration
from maxalaar_reinforcement_learning.configuration.video_episodes_generation_configuration import VideoEpisodesGenerationConfiguration


class ExperimentationConfiguration:
    """One experiment: an environment, how to train on it, and how to look at the trained agent.

    Projects that do more with the trained agents (analysis, datasets...) can subclass it and add their own
    fields; this package only reads the ones below.
    """

    def __init__(self, experimentation_name: str, environment_name: str = 'CartPole', environment_configuration=None):
        if environment_configuration is None:
            environment_configuration = {}

        self.wandb_api_key_file_path: Path = Path('./secrets/wandb_api_key.txt')

        self.experimentation_name: str = experimentation_name
        self.experimentation_storage_path: Path = Path.cwd() / 'experiments' / self.experimentation_name
        self.videos_path: Path = self.experimentation_storage_path / 'videos'

        self.environment_name: str = environment_name
        self.environment_configuration: dict = environment_configuration

        self.ray_local_mode: bool = False
        # mkdir -p /mnt/873a422e-1bf1-4c0b-b123-e47e6872b379/tmp/ray
        # ln -s /mnt/873a422e-1bf1-4c0b-b123-e47e6872b379/tmp/ray ~/ray_my_link
        self.ray_temporary_directory: Optional[Path | str] = None
        self.ray_bind_to_localhost: bool = True
        self.ray_health_check_period_ms: int = 5000
        self.ray_health_check_failure_threshold: int = 60
        self.ray_environment_variables: dict = {
            'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True',
            'MALLOC_ARENA_MAX': '1',
        }

        # Reinforcement Learning
        self.reinforcement_learning_storage_path: Path = self.experimentation_storage_path / 'reinforcement_learning'
        self.reinforcement_learning_configuration: Optional[ReinforcementLearningConfiguration] = ReinforcementLearningConfiguration()

        # Video Episodes
        self.video_episodes_storage_path: Path = self.videos_path / 'episodes'
        self.video_episodes_generation_configuration: Optional[VideoEpisodesGenerationConfiguration] = VideoEpisodesGenerationConfiguration()

        # Display Episodes
        self.display_episodes_configuration: Optional[DisplayEpisodesConfiguration] = DisplayEpisodesConfiguration()

    def environment_to_yaml_file(self, directory: Path):
        directory.mkdir(parents=True, exist_ok=True)
        file_path = directory / 'environment_configuration.yaml'
        environment_dictionary = {
            'environment_name': self.environment_name,
            'environment_configuration': self.environment_configuration,
        }
        with open(file_path, 'w') as file:
            yaml.dump(environment_dictionary, file, default_flow_style=False, allow_unicode=True)
