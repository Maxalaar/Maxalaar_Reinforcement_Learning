from pathlib import Path

def check_wandb_api_key_file(wandb_api_key_file_path: Path):
    if wandb_api_key_file_path.is_file():
        print(f"File {wandb_api_key_file_path} exists.")
    else:
        print(f"Warning: File {wandb_api_key_file_path} does not exist.")
        print("You can create it using the following command:")
        print("mkdir -p ./secrets && echo 'your_wandb_api_key' > ./secrets/wandb_api_key.txt")
