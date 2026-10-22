from pathlib import Path


class CurriculumConfiguration:
    """Experiments trained one after the other, each starting from the weights of the previous one.

    A stage is an experimentation configuration file; it ends when its stopping criterion is reached (every stage
    but the last needs one), and the next stage starts from its final checkpoint (weights only, a new optimizer).
    Training a curriculum again resumes it where it stopped (training/curriculum_training.py).
    """

    def __init__(self, curriculum_name: str, stages: list[str | Path]):
        if not stages:
            raise ValueError('a curriculum needs at least one stage')

        self.curriculum_name: str = curriculum_name
        # Experimentation configuration files, relative to the working directory (the project root).
        self.stages: list[Path] = [Path(stage) for stage in stages]
        # version_<k>/ holds one run of the whole curriculum: its state, its log and one directory per stage.
        self.curriculum_storage_path: Path = Path.cwd() / 'experiments' / curriculum_name
        # Replaces the stages' own frequency: an interrupted stage resumes from its last checkpoint.
        self.checkpoint_frequency: int = 5
