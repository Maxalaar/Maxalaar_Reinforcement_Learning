import argparse
import sys

from maxalaar_reinforcement_learning.training.curriculum_training import curriculum_training
from maxalaar_reinforcement_learning.utilities.save.get_configuration_class import get_configuration_class


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train the stages of a curriculum one after the other; run it again to resume where it stopped.'
    )
    parser.add_argument(
        '--curriculum_configuration_file',
        type=str,
        required=True,
        help="The path of the curriculum configuration file (a CurriculumConfiguration named like the file)"
    )
    parser.add_argument(
        '--new',
        action='store_true',
        help='Start a new run of the curriculum instead of resuming the latest one',
    )

    arguments = parser.parse_args()
    curriculum_configuration = get_configuration_class(arguments.curriculum_configuration_file)

    completed = curriculum_training(curriculum_configuration, new=arguments.new)
    sys.exit(0 if completed else 1)
