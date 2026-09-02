"""
Comprehensive Benchmark Script for Real-Time Face Swap Latency and Throughput.
Profiles execution performance across resolutions, providers, and pipeline stages.
"""

import os
import sys
import time
import csv
import argparse
from typing import List, Dict, Any
import numpy as np
import cv2

# Ensure repo root on path
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.core.config_loader import load_all_configs
from src.core.device import get_device_manager
from src.pipeline.realtime_pipeline import RealTimePipeline
from src.utils.logger import setup_logging, get_logger

logger = get_logger("Benchmark")


def run_benchmark(
    resolutions: List[tuple] = [(640, 480), (1280, 720), (1920, 1080)],
    iterations: int = 50,
    output_csv: str = "outputs/benchmarks/benchmark_results.csv",
) -> List[Dict[str, Any]]:
    """
    Executes throughput benchmark across specified resolutions.
    """
    setup_logging()
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    dm = get_device_manager()

    app_cfg, models_cfg, targets_cfg = load_all_configs()
    pipeline = RealTimePipeline(app_cfg, models_cfg, targets_cfg)

    # Select target if available
    targets = pipeline.target_manager.list_targets()
    if targets:
        pipeline.select_target(targets[0].target_id)

    results = []

    logger.info("=" * 60)
    logger.info(f"STARTING FACE SWAP BENCHMARK ({iterations} frames per resolution)")
    logger.info(f"Active Provider: {dm.active_provider}")
    logger.info("=" * 60)

    for w, h in resolutions:
        logger.info(f"\nBenchmarking Resolution: {w}x{h}...")

        # Create synthetic test frame with face
        test_frame = np.zeros((h, w, 3), dtype=np.uint8)
        test_frame[:] = (40, 40, 40)
        center_x, center_y = w // 2, h // 2
        radius = min(w, h) // 4
        cv2.circle(test_frame, (center_x, center_y), radius, (180, 200, 220), -1)

        # Warm-up pass
        for _ in range(5):
            pipeline.process_frame(test_frame)

        latencies = []
        swap_latencies = []

        for i in range(iterations):
            t0 = time.perf_counter()
            res = pipeline.process_frame(test_frame)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            latencies.append(elapsed_ms)
            swap_latencies.append(res.metrics_summary.get("swap_latency_ms", 0.0))

        avg_lat = sum(latencies) / len(latencies)
        min_lat = min(latencies)
        max_lat = max(latencies)
        avg_swap_lat = sum(swap_latencies) / len(swap_latencies)
        effective_fps = 1000.0 / avg_lat if avg_lat > 0 else 0.0

        row = {
            "resolution": f"{w}x{h}",
            "provider": dm.active_provider,
            "gpu_available": dm.is_gpu_available,
            "iterations": iterations,
            "avg_latency_ms": round(avg_lat, 2),
            "min_latency_ms": round(min_lat, 2),
            "max_latency_ms": round(max_lat, 2),
            "swap_latency_ms": round(avg_swap_lat, 2),
            "effective_fps": round(effective_fps, 1),
            "target_used": pipeline.get_selected_target().display_name if pipeline.get_selected_target() else "None",
        }
        results.append(row)

        logger.info(
            f"  {w}x{h} -> FPS: {row['effective_fps']} | "
            f"Avg Latency: {row['avg_latency_ms']} ms | "
            f"Min: {row['min_latency_ms']} ms | "
            f"Max: {row['max_latency_ms']} ms"
        )

    # Save to CSV
    fieldnames = [
        "resolution",
        "provider",
        "gpu_available",
        "iterations",
        "effective_fps",
        "avg_latency_ms",
        "min_latency_ms",
        "max_latency_ms",
        "swap_latency_ms",
        "target_used",
    ]
    try:
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        logger.info(f"\nSaved benchmark results to: '{output_csv}'")
    except Exception as e:
        logger.error(f"Failed to write benchmark CSV: {e}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark Face Swap Pipeline performance.")
    parser.add_argument("--iterations", "-n", type=int, default=30, help="Frames per resolution")
    parser.add_argument("--output", "-o", type=str, default="outputs/benchmarks/benchmark_results.csv", help="Output CSV")
    args = parser.parse_args()

    run_benchmark(iterations=args.iterations, output_csv=args.output)


if __name__ == "__main__":
    main()
