import os
import sys
import shutil
import kagglehub
import yaml


def load_config():
    with open("config/config.yaml", "r") as f:
        return yaml.safe_load(f)


def download_dataset():
    config = load_config()
    dataset_id = config["dataset"]["kaggle_id"]
    raw_path = config["dataset"]["raw_path"]

    print(f"Downloading dataset: {dataset_id}")
    print("This may take a while (~31 GB)...\n")

    # Download via kagglehub
    path = kagglehub.dataset_download(dataset_id)
    print(f"\nDownloaded to: {path}")

    # Copy to project directory
    os.makedirs(raw_path, exist_ok=True)

    if path != raw_path:
        print(f"Copying to project directory: {raw_path}")
        for item in os.listdir(path):
            src = os.path.join(path, item)
            dst = os.path.join(raw_path, item)
            if os.path.isdir(src):
                shutil.copytree(src, dst, dirs_exist_ok=True)
            else:
                shutil.copy2(src, dst)

    print("\nDataset ready!")
    print(f"Path: {raw_path}")

    # Print folder structure
    print("\nFolder structure:")
    for root, dirs, files in os.walk(raw_path):
        level = root.replace(raw_path, "").count(os.sep)
        indent = " " * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        if level < 2:  # Only show 2 levels deep
            sub_indent = " " * 2 * (level + 1)
            for file in files[:5]:
                print(f"{sub_indent}{file}")
            if len(files) > 5:
                print(f"{sub_indent}... and {len(files) - 5} more files")


if __name__ == "__main__":
    download_dataset()