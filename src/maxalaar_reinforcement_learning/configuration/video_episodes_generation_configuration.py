from ray.rllib.utils.from_config import NotProvided, _NotProvided


class VideoEpisodesGenerationConfiguration:
    def __init__(self):
        self.minimal_number_videos: int = 50
        self.environment_configuration: dict = {}
        self.number_environment_runners: int = 1
        self.number_environment_per_environment_runners: int = 1
        self.number_cpus_per_environment_runners: int | _NotProvided = NotProvided
        self.number_gpus_per_environment_runners: int | _NotProvided = NotProvided




