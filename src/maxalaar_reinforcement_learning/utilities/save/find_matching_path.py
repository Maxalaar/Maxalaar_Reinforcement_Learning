from pathlib import Path
from typing import List, Optional


from pathlib import Path
from typing import List

def find_matching_path(paths: List[Path], target_part: str) -> Path:
    """
    Find the unique path in a list that contains the target substring.

    Args:
        paths: List of paths.
        target_part: Substring to search for inside each path.

    Returns:
        The unique Path that contains the target substring.

    Raises:
        ValueError: If zero or multiple matching paths are found.
    """
    matching_paths = []

    for path_str in paths:
        path = Path(path_str)
        if target_part in path.as_posix():
            matching_paths.append(path)

    if len(matching_paths) == 0:
        raise ValueError(f"No path contains '{target_part}'.")
    if len(matching_paths) > 1:
        raise ValueError(
            f"Multiple paths contain '{target_part}': {matching_paths}"
        )

    return matching_paths[0]
