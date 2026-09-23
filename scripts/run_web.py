"""
Standalone Launcher Script for Real-Time AI Face Swap Web Studio.
Provides live interactive browser streaming and remote telemetry.
Usage:
    python scripts/run_web.py --port 8000
"""

import sys
import os
import argparse

# Ensure repository root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.web.app import run_web_studio
from src.utils.logger import setup_logging, get_logger

logger = get_logger("RunWeb")


def parse_args():
    parser = argparse.ArgumentParser(description="Real-Time AI Face Swap Web Studio")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host interface to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on (default: 8000)")
    parser.add_argument("--device", type=str, default="auto", help="Hardware device: auto, cuda, dml, cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging(log_file="logs/web_studio.log")

    logger.info("=" * 65)
    logger.info(f"Launching Real-Time AI Face Swap Web Studio on http://{args.host}:{args.port}")
    logger.info("=" * 65)

    run_web_studio(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
