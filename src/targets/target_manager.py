"""
Target Manager for Querying, Filtering, Selecting, and Dynamically Caching Target Faces.
"""

import os
import json
import time
from typing import Dict, List, Optional, Any, Tuple
import cv2
import numpy as np

from src.targets.target_loader import TargetFace, scan_target_database, TargetMetadata
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

    def get_all_targets(self) -> List[TargetFace]:
        """Alias for list_targets."""
        return self.list_targets()

    def reload_all_targets(self) -> None:
        """Alias for reload."""
        self.reload()

    def get_categories(self) -> List[str]:
        """Returns unique categories in the target database."""
        cats = set()
        for t in self._targets.values():
            if t.category:
                cats.add(t.category)
        return sorted(list(cats))

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

    def get_target_by_id(self, target_id: str) -> Optional[TargetFace]:
        """Alias for get_target."""
        return self.get_target(target_id)

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

    def add_custom_target(
        self,
        image_or_path: Any,
        display_name: str,
        category: str = "custom",
        person_id: Optional[str] = None,
    ) -> Tuple[bool, str, Optional[TargetFace]]:
        """
        Dynamically imports a new target face:
        Validates the face, extracts 512-D ArcFace embedding, saves reference.jpg,
        face.npy, and metadata.json under faces/<category>/<person_id>/, and registers it.
        """
        import os
        import json
        import time
        import cv2
        from src.targets.target_loader import TargetMetadata

        if isinstance(image_or_path, str):
            if not os.path.isfile(image_or_path):
                return False, f"Image file not found: {image_or_path}", None
            bgr = cv2.imread(image_or_path)
            if bgr is None:
                return False, f"Failed to decode image file: {image_or_path}", None
        elif isinstance(image_or_path, np.ndarray):
            bgr = image_or_path.copy()
        else:
            return False, f"Unsupported image input type: {type(image_or_path)}", None

        # Determine clean person_id
        if not person_id:
            safe_name = "".join(c if c.isalnum() else "_" for c in display_name.lower().strip())
            person_id = safe_name.strip("_")
            if not person_id:
                person_id = f"custom_{int(time.time())}"

        # Clean category
        safe_cat = "".join(c if c.isalnum() else "_" for c in category.lower().strip()).strip("_")
        if not safe_cat:
            safe_cat = "custom"

        # Compute embedding
        emb = self.extractor.extract_embedding(bgr)
        if emb is None or np.all(emb == 0):
            return False, "No valid face detected in reference image. Please ensure face is clearly visible.", None

        # Build directory path under faces/<category>/<person_id>
        target_dir = os.path.join(self.config.target_root, safe_cat, person_id)
        os.makedirs(target_dir, exist_ok=True)

        ref_path = os.path.join(target_dir, "reference.jpg")
        emb_path = os.path.join(target_dir, "face.npy")
        meta_path = os.path.join(target_dir, "metadata.json")

        # Save files
        cv2.imwrite(ref_path, bgr)
        np.save(emb_path, emb.astype(np.float32))

        meta_obj = TargetMetadata(
            person_id=person_id,
            display_name=display_name,
            category=safe_cat,
            source="custom_import",
            consent_status="user_provided",
        )
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_obj.to_dict(), f, indent=2)

        target_face = TargetFace(
            target_id=person_id,
            display_name=display_name,
            category=safe_cat,
            reference_image_path=ref_path,
            reference_image=bgr,
            embedding=emb.astype(np.float32),
            metadata=meta_obj,
        )

        self._targets[person_id] = target_face
        logger.info(f"Custom target '{display_name}' ({person_id}) registered successfully.")
        return True, f"Target '{display_name}' imported successfully!", target_face


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
