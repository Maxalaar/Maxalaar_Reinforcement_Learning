import os
import re
from pathlib import Path


def get_reinforcement_learning_version_number(storage_path: Path) -> int:
    """
    Creates a new versioned directory for training under the specified base path.

    Args:
        storage_path: The root directory where training runs are stored.

    Returns:
        str: The full path to the newly created version directory.
    """

    # Create the base training directory if it doesn't exist
    os.makedirs(storage_path, exist_ok=True)

    # Find existing version directories like 'version_0', 'version_1', ...
    existing_versions = [
        int(match.group(1))
        for name in os.listdir(storage_path)
        if os.path.isdir(os.path.join(storage_path, name)) and (match := re.match(r'version_(\d+)', name))
    ]

    return max(existing_versions) + 1 if existing_versions else 0

    # # Determine the next available version number
    # next_version = max(existing_versions) + 1 if existing_versions else 0
    # # version_dir = base_path / f'version_{next_version}'
    # #
    # # # Create the new version directory (fail if it already exists)
    # # os.makedirs(version_dir, exist_ok=False)
    #
    # return next_version
