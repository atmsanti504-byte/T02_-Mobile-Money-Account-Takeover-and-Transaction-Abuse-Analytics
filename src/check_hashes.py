import os
import hashlib
from pathlib import Path

def calculate_sha256(file_path):
    """Calculate the SHA-256 hash of a file."""
    sha256_hash = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            # Read the file in chunks to handle large files efficiently
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    except Exception as e:
        return f"Error reading file: {e}"

def generate_file_hashes():
    # Prompt the user to input the directory path dynamically
    target_dir = input("Enter the absolute path to the directory: ").strip()
    
    path = Path(target_dir)
    if not path.is_dir():
        print(f"Error: '{target_dir}' is not a valid directory.")
        return

    print(f"\nScanning directory: {path.resolve()}\n")
    print(f"{'File Path':<60} | {'SHA-256 Hash'}")
    print("-" * 130)

    # Walk through the directory and all subdirectories
    for root, _, files in os.walk(path):
        for file in files:
            file_path = Path(root) / file
            # Calculate relative path for cleaner output
            try:
                display_path = file_path.relative_to(path)
            except ValueError:
                display_path = file_path
                
            file_hash = calculate_sha256(file_path)
            print(f"{str(display_path):<60} | {file_hash}")

if __name__ == "__main__":
    generate_file_hashes()
