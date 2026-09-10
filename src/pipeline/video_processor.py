"""
Offline Video and Image Processing Engine.
Provides frame-accurate, synchronous batch face transformation for recorded videos and photos.
"""

import os
import sys
import time
import subprocess
import shutil
from typing import Optional, Callable, Dict, Any, List
import cv2
import numpy as np

from src.pipeline.realtime_pipeline import RealTimePipeline, PipelineResult
from src.core.config_loader import load_all_configs, AppConfig, ModelsConfig, TargetsConfig
from src.targets.target_loader import TargetFace
from src.utils.logger import get_logger

logger = get_logger("VideoProcessor")


class VideoFileProcessor:
    """
    Offline Media Transformation Engine.
    Executes frame-by-frame synchronous neural face swapping on video files and image assets.
    """

    def __init__(self, pipeline: Optional[RealTimePipeline] = None):
        if pipeline is not None:
            self.pipeline = pipeline
        else:
            a_cfg, m_cfg, t_cfg = load_all_configs()
            self.pipeline = RealTimePipeline(a_cfg, m_cfg, t_cfg)

    def process_image(
        self,
        input_image: Any,
        output_path: Optional[str] = None,
        target_id: Optional[str] = None,
        enhance_strength: Optional[float] = None,
    ) -> np.ndarray:
        """
        Transforms a single portrait photo or numpy BGR image.
        """
        if isinstance(input_image, str):
            if not os.path.isfile(input_image):
                raise FileNotFoundError(f"Input image not found: {input_image}")
            frame = cv2.imread(input_image)
            if frame is None:
                raise ValueError(f"Could not read image file: {input_image}")
        elif isinstance(input_image, np.ndarray):
            frame = input_image.copy()
        else:
            raise TypeError(f"Invalid input image type: {type(input_image)}")

        target: Optional[TargetFace] = None
        if target_id:
            self.pipeline.select_target(target_id)
            target = self.pipeline.get_selected_target()

        result: PipelineResult = self.pipeline.process_frame_sync(
            frame,
            target=target,
            enhance_strength=enhance_strength,
        )

        if output_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            cv2.imwrite(output_path, result.rendered_frame)
            logger.info(f"Saved swapped image to: {output_path}")

        return result.rendered_frame

    def process_directory(
        self,
        input_dir: str,
        output_dir: str,
        target_id: Optional[str] = None,
        enhance_strength: Optional[float] = None,
    ) -> List[str]:
        """
        Batch-swaps all supported images in a directory.
        """
        if not os.path.isdir(input_dir):
            raise NotADirectoryError(f"Directory not found: {input_dir}")

        os.makedirs(output_dir, exist_ok=True)
        valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        processed_files = []

        files = sorted(os.listdir(input_dir))
        for fname in files:
            ext = os.path.splitext(fname)[1].lower()
            if ext in valid_exts:
                in_path = os.path.join(input_dir, fname)
                out_path = os.path.join(output_dir, f"swapped_{fname}")
                try:
                    self.process_image(in_path, out_path, target_id, enhance_strength)
                    processed_files.append(out_path)
                except Exception as e:
                    logger.error(f"Failed to process image '{fname}': {e}")

        logger.info(f"Batch processed {len(processed_files)} images into: {output_dir}")
        return processed_files

    def process_video(
        self,
        input_path: str,
        output_path: str,
        target_id: Optional[str] = None,
        enhance_strength: Optional[float] = None,
        progress_callback: Optional[Callable[[int, int, float, float], None]] = None,
    ) -> Dict[str, Any]:
        """
        Transforms a video file frame-by-frame and muxes audio from source.
        Args:
            input_path: Path to source video file (.mp4, .avi, .mov).
            output_path: Target path for output video file.
            target_id: Target face ID to swap into.
            enhance_strength: Face enhancement strength [0.0, 1.0].
            progress_callback: Optional callable(curr_frame, total_frames, fps, eta_seconds).
        Returns:
            Dict containing processing metrics and file paths.
        """
        if not os.path.isfile(input_path):
            raise FileNotFoundError(f"Input video file not found: {input_path}")

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise IOError(f"Could not open video file: {input_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if fps <= 0 or np.isnan(fps):
            fps = 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        logger.info(f"Ingesting video: {input_path} ({width}x{height}, {fps:.1f} FPS, {total_frames} frames)")

        # Target selection
        target: Optional[TargetFace] = None
        if target_id:
            self.pipeline.select_target(target_id)
            target = self.pipeline.get_selected_target()

        # Temporary video output before audio muxing
        temp_video = output_path + ".temp.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(temp_video, fourcc, fps, (width, height))
        if not writer.isOpened():
            cap.release()
            raise IOError(f"Could not create video writer for: {temp_video}")

        curr_frame = 0
        swapped_frames = 0
        t_start = time.perf_counter()

        try:
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                curr_frame += 1
                result: PipelineResult = self.pipeline.process_frame_sync(
                    frame,
                    target=target,
                    enhance_strength=enhance_strength,
                )

                if result.is_swapped:
                    swapped_frames += 1

                writer.write(result.rendered_frame)

                # Telemetry
                elapsed = time.perf_counter() - t_start
                proc_fps = curr_frame / max(0.001, elapsed)
                rem_frames = max(0, total_frames - curr_frame)
                eta_sec = rem_frames / max(0.001, proc_fps)

                if progress_callback is not None:
                    progress_callback(curr_frame, total_frames, proc_fps, eta_sec)

                if curr_frame % 50 == 0 or curr_frame == total_frames:
                    logger.info(
                        f"Processing video: {curr_frame}/{total_frames} "
                        f"({(curr_frame/max(1, total_frames))*100:.1f}%) | "
                        f"{proc_fps:.1f} FPS | ETA: {eta_sec:.1f}s"
                    )

        finally:
            cap.release()
            writer.release()

        # Audio Muxing
        has_audio_muxed = self._mux_audio_track(input_path, temp_video, output_path)
        if not has_audio_muxed:
            # Fallback: rename temp video to output path
            if os.path.exists(output_path):
                os.remove(output_path)
            shutil.move(temp_video, output_path)

        # Cleanup temp file if still present
        if os.path.exists(temp_video):
            try:
                os.remove(temp_video)
            except OSError:
                pass

        total_elapsed = time.perf_counter() - t_start
        summary = {
            "input_path": input_path,
            "output_path": output_path,
            "total_frames": curr_frame,
            "swapped_frames": swapped_frames,
            "fps_processed": round(curr_frame / max(0.001, total_elapsed), 2),
            "elapsed_seconds": round(total_elapsed, 2),
            "audio_preserved": has_audio_muxed,
        }
        logger.info(f"Video processing finished: {output_path} in {total_elapsed:.1f}s")
        return summary

    def _mux_audio_track(self, source_video: str, swapped_video: str, final_output: str) -> bool:
        """
        Extracts original audio track from source_video and muxes into swapped_video using ffmpeg.
        """
        # Check if ffmpeg is available in system PATH
        ffmpeg_cmd = shutil.which("ffmpeg")
        if not ffmpeg_cmd:
            return False

        try:
            cmd = [
                ffmpeg_cmd,
                "-y",
                "-i", swapped_video,
                "-i", source_video,
                "-c:v", "copy",
                "-c:a", "aac",
                "-map", "0:v:0",
                "-map", "1:a:0?",
                "-shortest",
                final_output,
            ]
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
            if result.returncode == 0 and os.path.isfile(final_output) and os.path.getsize(final_output) > 0:
                return True
        except Exception as e:
            logger.warning(f"FFmpeg audio muxing exception: {e}")

        return False
