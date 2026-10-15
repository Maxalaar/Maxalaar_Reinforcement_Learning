import json
import os
from pathlib import Path


def save_dictionary(dictionary: dict, name: str, path: Path):
    """
    Saves a dictionary as a readable JSON file to the specified path with the given file name.

    :param dictionary: The dictionary to save.
    :param name: The name of the file to save the dictionary to.
    :param path: The directory path where the dictionary should be saved.
    """
    try:
        # Create the full file path
        file_path = path / name
        file_path = file_path.with_suffix('.json')

        # Check if the directory exists, if not, create it
        os.makedirs(path, exist_ok=True)

        # If the file exists, remove it
        if file_path.exists():
            file_path.unlink()

        # Open the file in write mode and save the dictionary as formatted JSON
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(dictionary, file, ensure_ascii=False, indent=4)

        print(f"Dictionary successfully saved to {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")