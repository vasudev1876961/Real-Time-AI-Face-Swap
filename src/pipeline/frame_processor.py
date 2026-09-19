"""
Frame Processor and Batch Execution Bridge.
"""

from src.pipeline.realtime_pipeline import RealTimePipeline, PipelineResult
from src.pipeline.video_processor import VideoFileProcessor

__all__ = ["RealTimePipeline", "PipelineResult", "VideoFileProcessor"]
