from pathlib import Path

def save_text(text: str, file_path: Path) -> None:
    """
    Saves a string to a text file. Ensures the file has a '.txt' extension.

    Args:
        text (str): The text to save.
        file_path (Path): The path to the file (with or without .txt extension).
    """
    file_path = file_path.with_suffix('.txt')  # Force l'extension .txt

    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(text)
    print(f"The text has been saved to {file_path}")
