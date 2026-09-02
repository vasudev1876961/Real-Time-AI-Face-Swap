"""
Target Manager for Querying, Filtering, Selecting, and Dynamically Caching Target Faces.
"""

from typing import Dict, List, Optional
from src.targets.target_loader import TargetFace, scan_target_database
from src.targets.target_embedding import TargetEmbeddingExtractor
from src.core.config_loader import TargetsConfig
from src.utils.logger import get_logger

logger = get_logger("TargetManager")


class TargetManager:
    """
    Coordinates target face database operations with thread-safe cached access.
    """

    def __init__(
        self,
        config: Optional[TargetsConfig] = None,
        embedding_extractor: Optional[TargetEmbeddingExtractor] = None,
    ):
        self.config = config or TargetsConfig()
        self.extractor = embedding_extractor or TargetEmbeddingExtractor()
        self._targets: Dict[str, TargetFace] = {}
        self._selected_target_id: Optional[str] = None
        self.reload()

    def reload(self) -> None:
        """Rescans the filesystem target database and updates in-memory cache."""
        self._targets = scan_target_database(
            target_root=self.config.target_root,
            embedding_extractor=self.extractor,
        )
        if self._selected_target_id and self._selected_target_id not in self._targets:
            self._selected_target_id = None

    def list_targets(self) -> List[TargetFace]:
        """Returns all loaded targets."""
        return list(self._targets.values())

    def list_by_category(self, category: str) -> List[TargetFace]:
        """
        Returns all targets matching the given category ID.
        If category is 'all' or empty, returns all targets.
        """
        cat_lower = category.strip().lower()
        if cat_lower in ["all", "", "none"]:
            return list(self._targets.values())
        return [t for t in self._targets.values() if t.category.lower() == cat_lower]

    def get_target(self, target_id: str) -> Optional[TargetFace]:
        """Fetches target by unique ID."""
        return self._targets.get(target_id)

    def select_target(self, target_id: Optional[str]) -> bool:
        """
        Selects an active target for face swapping.
        Pass None or empty string to deselect.
        """
        if target_id is None or target_id == "":
            self._selected_target_id = None
            logger.info("Target face deselected.")
            return True

        if target_id in self._targets:
            self._selected_target_id = target_id
            target = self._targets[target_id]
            logger.info(f"Target selected: '{target.display_name}' (ID: {target_id}, Category: {target.category})")
            return True
        else:
            logger.warning(f"Target ID '{target_id}' not found in target database.")
            return False

    def get_selected_target(self) -> Optional[TargetFace]:
        """Returns the currently selected TargetFace or None."""
        if self._selected_target_id is None:
            return None
        return self._targets.get(self._selected_target_id)


_GLOBAL_TARGET_MANAGER: Optional[TargetManager] = None


def get_target_manager(
    config: Optional[TargetsConfig] = None,
    embedding_extractor: Optional[TargetEmbeddingExtractor] = None,
) -> TargetManager:
    """Returns singleton TargetManager instance."""
    global _GLOBAL_TARGET_MANAGER
    if _GLOBAL_TARGET_MANAGER is None:
        _GLOBAL_TARGET_MANAGER = TargetManager(config, embedding_extractor)
    return _GLOBAL_TARGET_MANAGER
