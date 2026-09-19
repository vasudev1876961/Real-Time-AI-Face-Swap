"""
Post-Processing Module (Re-export and Interface Adapter).
Bridges to src.processing.postprocess for unsharp sharpening and telemetry rendering.
"""

from src.processing.postprocess import (
    postprocess_frame,
    apply_unsharp_mask,
    draw_telemetry_hud,
)

__all__ = [
    "postprocess_frame",
    "apply_unsharp_mask",
    "draw_telemetry_hud",
]
