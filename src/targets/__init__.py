"""
Target Face Management and Embedding Subsystem.
"""

from src.targets.target_embedding import TargetEmbeddingExtractor
from src.targets.target_loader import TargetFace, TargetMetadata, load_target_from_dir, scan_target_database
from src.targets.target_manager import TargetManager, get_target_manager

__all__ = [
    "TargetFace",
    "TargetMetadata",
    "TargetEmbeddingExtractor",
    "load_target_from_dir",
    "scan_target_database",
    "TargetManager",
    "get_target_manager",
]
