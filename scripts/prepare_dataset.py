"""
Dataset Preparation CLI Launcher.
"""

import sys
import os

# Ensure repo root is on sys.path
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from dataset_tools.prepare_dataset import main

if __name__ == "__main__":
    main()
