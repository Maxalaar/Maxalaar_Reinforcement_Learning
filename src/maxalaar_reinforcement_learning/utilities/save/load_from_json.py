import json
from pathlib import Path


def load_from_json(file_path: Path) -> dict:
    """
    Load game scene descriptions from a JSON file and return them as a Python dictionary.

    Args:
        file_path (str): Path to the JSON file containing the game scene descriptions.

    Returns:
        dict: A dictionary where keys are scene IDs and values are scene descriptions.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        scenes_dict = json.load(file)
    return scenes_dict

