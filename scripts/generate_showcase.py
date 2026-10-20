"""
Standalone Showcase & Visual Output Demonstration Generator (Phase 17).
Generates high-resolution comparative grids, split-screen wipes, and studio master outputs.
Usage:
    python scripts/generate_showcase.py --target prabhas
"""

import os
import sys
import argparse
import shutil
import cv2

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.processing.showcase_generator import ShowcaseGenerator
from src.core.config_loader import load_all_configs
from src.utils.logger import setup_logging, get_logger

logger = get_logger("RunShowcase")


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 17 Output Showcase Generator")
    parser.add_argument("--input", type=str, default="", help="Path to input portrait/frame (optional)")
    parser.add_argument("--target", type=str, default="prabhas", help="Target identity ID (default: prabhas)")
    parser.add_argument("--split-target", type=str, default="rekha", help="Secondary/alternative target ID")
    parser.add_argument("--outdir", type=str, default="outputs", help="Output directory")
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging(log_file="logs/showcase.log")

    logger.info("=" * 65)
    logger.info("Generating Real-Time AI Face Swap Phase 17 Showcase Demonstrations")
    logger.info(f"Target Identity: {args.target}")
    logger.info("=" * 65)

    os.makedirs(args.outdir, exist_ok=True)

    # Resolve input image
    input_path = args.input
    if not input_path or not os.path.isfile(input_path):
        sample_candidates = [
            os.path.join(PROJECT_ROOT, "outputs", "captures", "capture_2026-08-24_21-47-58_hero_001.jpg"),
            os.path.join(PROJECT_ROOT, "outputs", "captures", "test_swapped_phase7.jpg"),
            os.path.join(PROJECT_ROOT, "outputs", "captures", "test_phase9_rekha_prabhas.jpg"),
        ]
        for c in sample_candidates:
            if os.path.isfile(c):
                input_path = c
                break

    if not input_path or not os.path.isfile(input_path):
        logger.error("No valid input portrait found to generate showcase.")
        return

    logger.info(f"Using input source image: {input_path}")
    frame = cv2.imread(input_path)
    if frame is None:
        logger.error(f"Failed to read image at {input_path}")
        return

    # Initialize generator
    generator = ShowcaseGenerator()

    # 1. Generate 4-Stage Comparison Grid
    grid_out = os.path.join(args.outdir, "showcase_comparison_grid.jpg")
    generator.generate_comparison_grid(frame, target_id=args.target, output_path=grid_out)
    logger.info(f"Generated 4-Stage Showcase Grid -> {grid_out}")

    # 2. Generate Split-Screen Vertical Wipe
    split_out = os.path.join(args.outdir, "showcase_split_wipe.jpg")
    generator.generate_split_showcase(frame, target_id=args.target, output_path=split_out, split_pos=0.50)
    logger.info(f"Generated Split Wipe Showcase -> {split_out}")

    # 3. Generate Studio Master (Full Phase 17 Output)
    studio_out = os.path.join(args.outdir, "showcase_phase17_studio.jpg")
    generator.pipeline.select_target(args.target)
    generator.pipeline.set_volumetric_relighting_config(enabled=True, shadow_strength=0.50, sss_warmth=0.45)
    generator.pipeline.set_cinematic_bokeh_config(enabled=True, bokeh_radius=12.0, focus_depth=0.35)
    generator.pipeline.set_studio_lighting_config(enabled=True, rim_intensity=0.60, rim_color="warm_gold", backdrop_dim=0.15)
    generator.pipeline.set_temporal_coherence_config(enabled=True, smoothing_factor=0.85)

    res = None
    for _ in range(15):
        res = generator.pipeline.process_frame(frame.copy())
        if res.is_swapped:
            break
    if res and res.rendered_frame is not None:
        cv2.imwrite(studio_out, res.rendered_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        logger.info(f"Generated Phase 17 Studio Master Image -> {studio_out}")

    logger.info("=" * 65)
    logger.info("Showcase Generation Complete!")
    logger.info(f"View outputs at: {os.path.abspath(args.outdir)}")
    logger.info("=" * 65)


if __name__ == "__main__":
    main()
