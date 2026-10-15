import os
import shutil
from pathlib import Path


def create_directory(path: Path):
    if path.exists() and path.is_dir():
        shutil.rmtree(path)

    os.makedirs(path, exist_ok=True)