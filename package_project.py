"""
Packaging and Verification Script for Handwritten Digit Classifier AI
Creates a clean, portable distribution ZIP excluding cache and temporary files,
and verifies ZIP integrity and extraction.
"""

import os
import sys
import zipfile
import tempfile
import hashlib

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
DIST_DIR = os.path.join(PROJECT_ROOT, "dist")
ZIP_PATH = os.path.join(DIST_DIR, "Handwritten-Digit-Classifier-AI.zip")

# Items to include in the package
INCLUDES = [
    "app.py",
    "train.py",
    "predict.py",
    "package_project.py",
    "requirements.txt",
    "README.md",
    "run.bat",
    ".gitignore",
    "src",
    "tests",
    "models",
    "reports",
    "assets",
    "data",
    "demo_snapshots"
]

# Patterns/directories to exclude
EXCLUDES = {
    "__pycache__",
    ".pytest_cache",
    ".venv",
    "venv",
    "env",
    ".git",
    "dist",
    ".DS_Store",
    "Thumbs.db",
    "*.pyc",
    "*.pyo",
    "*.tmp",
    "*.log"
}


def should_exclude(rel_path: str) -> bool:
    parts = rel_path.replace("\\", "/").split("/")
    for part in parts:
        if part in EXCLUDES:
            return True
        if part.endswith(".pyc") or part.endswith(".tmp") or part.endswith(".log"):
            return True
    return False


def build_zip():
    os.makedirs(DIST_DIR, exist_ok=True)
    if os.path.exists(ZIP_PATH):
        os.remove(ZIP_PATH)

    print(f"Creating distribution package at:\n  {ZIP_PATH}")
    file_count = 0

    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for item in INCLUDES:
            item_path = os.path.join(PROJECT_ROOT, item)
            if not os.path.exists(item_path):
                continue

            if os.path.isfile(item_path):
                rel = os.path.relpath(item_path, PROJECT_ROOT)
                if not should_exclude(rel):
                    zip_file.write(item_path, rel)
                    file_count += 1
            elif os.path.isdir(item_path):
                for root, dirs, files in os.walk(item_path):
                    dirs[:] = [d for d in dirs if not should_exclude(d)]
                    for file in files:
                        full_path = os.path.join(root, file)
                        rel = os.path.relpath(full_path, PROJECT_ROOT)
                        if not should_exclude(rel):
                            zip_file.write(full_path, rel)
                            file_count += 1

    size_mb = os.path.getsize(ZIP_PATH) / (1024 * 1024)
    print(f"Package successfully created ({file_count} files, {size_mb:.2f} MB).")

    # Compute SHA-256
    hasher = hashlib.sha256()
    with open(ZIP_PATH, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    sha256_hash = hasher.hexdigest()
    print(f"SHA-256: {sha256_hash}")

    # Verify ZIP integrity
    print("\nVerifying ZIP integrity...")
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        bad_file = z.testzip()
        if bad_file is not None:
            raise RuntimeError(f"Corrupted file inside ZIP: {bad_file}")
        namelist = z.namelist()
        print(f"ZIP structure valid. Total verified files: {len(namelist)}")

    # Test test extraction into a temporary folder
    with tempfile.TemporaryDirectory() as tmp_dir:
        print(f"Testing extraction into temporary directory: {tmp_dir}")
        with zipfile.ZipFile(ZIP_PATH, "r") as z:
            z.extractall(tmp_dir)
        
        # Verify critical entrypoints exist
        for key_file in ["app.py", "train.py", "predict.py", "requirements.txt", "run.bat"]:
            assert os.path.exists(os.path.join(tmp_dir, key_file)), f"Missing {key_file} in extraction"
        print("Extraction test passed successfully!")

    print("\nPackaging and integrity verification completed with 0 errors.")


if __name__ == "__main__":
    build_zip()

