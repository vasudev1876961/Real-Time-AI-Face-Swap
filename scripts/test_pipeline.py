"""
Integration test script for Real-Time Face Swapping Pipeline.
Tests face detection, landmark alignment, ONNX swap with emap, and Lanczos4 blending.
"""

import os
import sys
import time
import cv2
import numpy as np

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.pipeline.realtime_pipeline import RealTimePipeline
from src.core.config_loader import load_all_configs


def test_pipeline_pass():
    print("=" * 65)
    print("Running End-to-End Real-Time Pipeline Integration Test")
    print("=" * 65)

    a_cfg, m_cfg, t_cfg = load_all_configs()
    # Ensure clarity sharpening is enabled
    a_cfg.processing.postprocess_sharpen = 0.40

    pipeline = RealTimePipeline(a_cfg, m_cfg, t_cfg)

    # Load test image
    test_img_path = "outputs/captures/capture_2026-08-24_21-47-58_hero_001.jpg"
    if not os.path.isfile(test_img_path):
        print(f"[ERROR] Test image not found at {test_img_path}")
        return

    frame = cv2.imread(test_img_path)
    print(f"Loaded test frame: {frame.shape}")

    # Select target: prabhas
    target_id = "prabhas"
    success = pipeline.select_target(target_id)
    print(f"Select target '{target_id}': {success}")

    # Process passes (wait for async background worker to complete inference)
    print("Executing pipeline passes...")
    max_wait = 10.0
    start = time.time()
    pass_count = 0
    while time.time() - start < max_wait:
        pass_count += 1
        result = pipeline.process_frame(frame.copy())
        if result.is_swapped:
            print(f"  Pass {pass_count}: status='{result.status_message}', is_swapped=True (took {time.time()-start:.2f}s)")
            break
        time.sleep(0.2)

    out_file = "outputs/test_pipeline_output.jpg"
    cv2.imwrite(out_file, result.rendered_frame)
    print(f"\nSaved final swapped result to {out_file}")
    print("=" * 65)


if __name__ == "__main__":
    test_pipeline_pass()
