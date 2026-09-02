"""
Launcher Script for the Real-Time Face Swap Camera Application.
"""

import sys
import os

# Ensure repository root is on sys.path
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.main import main

if __name__ == "__main__":
    sys.exit(main())
