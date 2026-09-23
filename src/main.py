"""
Application Entrypoint for Real-Time AI Face Swap Camera.
"""

import sys
import os

# Ensure workspace root is always on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import argparse
from PyQt6.QtWidgets import QApplication

from ui.main_window import MainWindow
from src.core.config_loader import load_all_configs
from src.core.device import get_device_manager
from src.utils.logger import setup_logging, get_logger

logger = get_logger("Main")


def parse_args():
    parser = argparse.ArgumentParser(description="Real-Time AI Face Swap Camera Application")
    parser.add_argument("--config-dir", type=str, default="configs", help="Path to configs directory")
    parser.add_argument("--camera", type=int, default=None, help="Override camera device index")
    parser.add_argument("--device", type=str, default="auto", help="Hardware device: auto, cuda, dml, cpu")
    parser.add_argument("--headless-check", action="store_true", help="Perform smoke test initialization without GUI")
    parser.add_argument("--web", action="store_true", help="Launch interactive browser Web Studio")
    parser.add_argument("--web-port", type=int, default=8000, help="Web Studio port (default: 8000)")
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging(log_file="logs/app.log")

    logger.info("=" * 60)
    logger.info("Starting Real-Time AI Face Swap Camera Application")
    logger.info("=" * 60)

    # Initialize device manager & check execution providers
    device_manager = get_device_manager(preferred_device=args.device)
    app_cfg, models_cfg, targets_cfg = load_all_configs(args.config_dir)

    if args.camera is not None:
        app_cfg.camera.camera_index = args.camera

    if args.web:
        logger.info(f"Launching Web Studio on port {args.web_port}...")
        from src.web.app import run_web_studio
        run_web_studio(port=args.web_port)
        return 0

    if args.headless_check:
        logger.info("Executing headless verification check...")
        from src.pipeline.realtime_pipeline import RealTimePipeline
        pipeline = RealTimePipeline(app_cfg, models_cfg, targets_cfg)
        status = pipeline.model_manager.get_status_summary()
        logger.info(f"Headless check completed. Subsystems status: {status}")
        return 0

    # Launch GUI Application
    app = QApplication(sys.argv)
    app.setApplicationName("Real-Time AI Face Swap Camera")
    app.setOrganizationName("AILabs")

    window = MainWindow(app_cfg, models_cfg, targets_cfg)
    window.show()

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
