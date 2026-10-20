"""
Interactive Output Showcase Generator (Phase 17).
Produces high-resolution side-by-side comparison grids, split-screen wipes,
and studio demonstration images across all pipeline stages.
"""

import os
import time
from typing import Dict, Any, List, Optional, Tuple
import cv2
import numpy as np

from src.core.config_loader import load_all_configs
from src.pipeline.realtime_pipeline import RealTimePipeline
from src.processing.split_screen import SplitScreenRenderer
from src.utils.logger import get_logger

logger = get_logger("ShowcaseGenerator")


class ShowcaseGenerator:
    """
    Automated visual benchmark and demonstration showcase generator.
    Produces high-fidelity comparison assets displaying the progression from
    original camera feed to Phase 17 studio master output.
    """

    def __init__(self, pipeline: Optional[RealTimePipeline] = None):
        if pipeline is not None:
            self.pipeline = pipeline
        else:
            a_cfg, m_cfg, t_cfg = load_all_configs()
            self.pipeline = RealTimePipeline(a_cfg, m_cfg, t_cfg)

    def generate_comparison_grid(
        self,
        input_frame: np.ndarray,
        target_id: str = "prabhas",
        output_path: Optional[str] = None,
    ) -> np.ndarray:
        """
        Executes a 4-stage comparative grid:
        [1. Original Camera] | [2. Raw Neural Swap]
        [3. Cinematic Bokeh + Relit] | [4. Phase 17 Studio Master]
        """
        if input_frame is None or input_frame.size == 0:
            raise ValueError("Input frame is empty.")

        h, w = input_frame.shape[:2]
        self.pipeline.select_target(target_id)

        # Stage 1: Original
        stage1 = input_frame.copy()
        cv2.putText(
            stage1, "1. ORIGINAL CAMERA", (24, 45),
            cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 0, 0), 3, cv2.LINE_AA
        )
        cv2.putText(
            stage1, "1. ORIGINAL CAMERA", (24, 45),
            cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 1, cv2.LINE_AA
        )

        # Helper to wait for async worker
        def run_pipeline_passes(max_passes: int = 15) -> np.ndarray:
            res = None
            for _ in range(max_passes):
                res = self.pipeline.process_frame(input_frame.copy())
                if res.is_swapped:
                    break
                time.sleep(0.04)
            return res.rendered_frame if res else input_frame.copy()

        # Stage 2: Standard Raw Swap (disable bokeh, relighting, studio lighting)
        self.pipeline.set_cinematic_bokeh_config(enabled=False)
        self.pipeline.set_volumetric_relighting_config(enabled=False)
        self.pipeline.set_micro_expression_config(enabled=False)
        self.pipeline.set_studio_lighting_config(enabled=False)
        stage2 = run_pipeline_passes()
        cv2.putText(
            stage2, "2. BASE NEURAL SWAP", (24, 45),
            cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 0, 0), 3, cv2.LINE_AA
        )
        cv2.putText(
            stage2, "2. BASE NEURAL SWAP", (24, 45),
            cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 220, 100), 1, cv2.LINE_AA
        )

        # Stage 3: Phase 15 & 16 (Volumetric Relighting & Cinematic Bokeh)
        self.pipeline.set_volumetric_relighting_config(enabled=True, shadow_strength=0.55, sss_warmth=0.45)
        self.pipeline.set_cinematic_bokeh_config(enabled=True, bokeh_radius=14.0, focus_depth=0.35)
        self.pipeline.set_micro_expression_config(enabled=True, strength=0.65)
        self.pipeline.set_studio_lighting_config(enabled=False)
        stage3 = run_pipeline_passes()
        cv2.putText(
            stage3, "3. RELIT + CINEMATIC BOKEH", (24, 45),
            cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 0, 0), 3, cv2.LINE_AA
        )
        cv2.putText(
            stage3, "3. RELIT + CINEMATIC BOKEH", (24, 45),
            cv2.FONT_HERSHEY_DUPLEX, 0.9, (100, 255, 220), 1, cv2.LINE_AA
        )

        # Stage 4: Phase 17 Studio Master (Virtual Rim Light + Temporal Coherence + Studio Grade)
        self.pipeline.set_studio_lighting_config(
            enabled=True,
            rim_intensity=0.60,
            rim_thickness=4,
            rim_color="warm_gold",
            backdrop_dim=0.15,
        )
        self.pipeline.set_temporal_coherence_config(enabled=True, smoothing_factor=0.85)
        stage4 = run_pipeline_passes()
        cv2.putText(
            stage4, "4. PHASE 17 STUDIO MASTER", (24, 45),
            cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 0, 0), 3, cv2.LINE_AA
        )
        cv2.putText(
            stage4, "4. PHASE 17 STUDIO MASTER", (24, 45),
            cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 180, 50), 1, cv2.LINE_AA
        )

        # Stitch into 2x2 grid
        half_w, half_h = w // 2, h // 2
        r1 = cv2.resize(stage1, (half_w, half_h))
        r2 = cv2.resize(stage2, (half_w, half_h))
        r3 = cv2.resize(stage3, (half_w, half_h))
        r4 = cv2.resize(stage4, (half_w, half_h))

        top_row = np.hstack([r1, r2])
        bot_row = np.hstack([r3, r4])
        grid = np.vstack([top_row, bot_row])

        # Add center watermark bar
        hdr_bar = np.zeros((46, grid.shape[1], 3), dtype=np.uint8)
        hdr_bar[:] = (20, 20, 25)
        title_text = f"REAL-TIME AI FACE SWAP — PHASE 17 STUDIO MASTER (TARGET: {target_id.upper()})"
        cv2.putText(
            hdr_bar, title_text, (20, 30),
            cv2.FONT_HERSHEY_DUPLEX, 0.75, (230, 230, 240), 1, cv2.LINE_AA
        )
        grid = np.vstack([hdr_bar, grid])

        if output_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            cv2.imwrite(output_path, grid, [cv2.IMWRITE_JPEG_QUALITY, 95])
            logger.info(f"Saved comparison grid showcase to {output_path}")

        return grid

    def generate_split_showcase(
        self,
        input_frame: np.ndarray,
        target_id: str = "prabhas",
        output_path: Optional[str] = None,
        split_pos: float = 0.50,
    ) -> np.ndarray:
        """Generates a split-screen comparative wipe with glowing divider."""
        self.pipeline.select_target(target_id)
        self.pipeline.set_volumetric_relighting_config(enabled=True)
        self.pipeline.set_studio_lighting_config(enabled=True, rim_intensity=0.55, rim_color="warm_gold")
        self.pipeline.set_split_screen_config(mode="split_vertical", position=split_pos)

        res = None
        for _ in range(15):
            res = self.pipeline.process_frame(input_frame.copy())
            if res.is_swapped:
                break
            time.sleep(0.04)

        out_frame = res.rendered_frame if res else input_frame.copy()
        # Reset split screen to off
        self.pipeline.set_split_screen_config(mode="off")

        if output_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            cv2.imwrite(output_path, out_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            logger.info(f"Saved split showcase to {output_path}")

        return out_frame
