"""
Dynamic Speech Articulation, Smile Energy, and Facial Expression Transfer Engine.
Transfers authentic speaker speech phoneme dynamics, mouth corner smile elevation,
and eyebrow expressivity from the original camera feed onto the synthetic face.
"""

from typing import Tuple, Optional, Dict, Any
import cv2
import numpy as np

from src.detection.face_landmarks import INSWAPPER_STANDARD_128
from src.utils.logger import get_logger

logger = get_logger("ExpressionTransfer")


class SpeechExpressionTransferEngine:
    """
    Real-time speech articulation and facial expression transfer engine.
    Extracts dynamic speech aperture, mouth corner elevation, and eyebrow expressivity
    from the original camera face crop and harmonizes it onto the swapped synthetic identity.
    """

    def __init__(
        self,
        default_strength: float = 0.65,
        smoothing_alpha: float = 0.70,
        enable_speech_aperture: bool = True,
        enable_smile_transfer: bool = True,
        enable_brow_dynamics: bool = True,
    ):
        self.default_strength = float(default_strength)
        self.smoothing_alpha = float(smoothing_alpha)
        self.enable_speech_aperture = bool(enable_speech_aperture)
        self.enable_smile_transfer = bool(enable_smile_transfer)
        self.enable_brow_dynamics = bool(enable_brow_dynamics)

        # Per-track temporal smoothing states
        # track_id -> dict of smoothed features
        self._states: Dict[int, Dict[str, float]] = {}

    def reset(self, track_id: Optional[int] = None) -> None:
        """Resets smoothed states."""
        if track_id is not None:
            self._states.pop(track_id, None)
        else:
            self._states.clear()

    def transfer_expressions(
        self,
        original_crop: np.ndarray,
        swapped_crop: np.ndarray,
        landmarks: Optional[np.ndarray] = None,
        strength: Optional[float] = None,
        track_id: int = 1,
    ) -> np.ndarray:
        """
        Dynamically transfers authentic speech aperture and expression details from original_crop
        to swapped_crop.

        Args:
            original_crop: (128, 128, 3) or (512, 512, 3) source camera face.
            swapped_crop: (128, 128, 3) or (512, 512, 3) neural swapped face.
            landmarks: 5 standard landmarks in crop coordinates (defaults to INSWAPPER_STANDARD_128).
            strength: Modulation factor [0.0, 1.0].
            track_id: Track ID for temporal smoothing.

        Returns:
            Expressively harmonized swapped face crop.
        """
        effective_strength = float(strength if strength is not None else self.default_strength)
        if effective_strength <= 0.01:
            return swapped_crop

        h, w = swapped_crop.shape[:2]
        if original_crop.shape[:2] != (h, w):
            orig = cv2.resize(original_crop, (w, h), interpolation=cv2.INTER_LANCZOS4)
        else:
            orig = original_crop

        scale = h / 128.0
        lms = (landmarks if landmarks is not None else INSWAPPER_STANDARD_128) * scale

        # Retrieve or initialize track state
        state = self._states.setdefault(
            track_id,
            {"speech_aperture": None, "smile_energy": None, "brow_shift": None},
        )

        result = swapped_crop.copy()

        # 1. Speech Phoneme Articulation (Inter-Labial Aperture & Oral Cavity Motion)
        if self.enable_speech_aperture:
            result = self._transfer_speech_aperture(orig, result, lms, state, scale, effective_strength)

        # 2. Smile Energy & Mouth Corner Elevation
        if self.enable_smile_transfer:
            result = self._transfer_smile_energy(orig, result, lms, state, scale, effective_strength)

        # 3. Eyebrow Dynamics & Expressivity
        if self.enable_brow_dynamics:
            result = self._transfer_brow_dynamics(orig, result, lms, state, scale, effective_strength)

        return result

    def _transfer_speech_aperture(
        self,
        orig: np.ndarray,
        swapped: np.ndarray,
        lms: np.ndarray,
        state: Dict[str, Any],
        scale: float,
        strength: float,
    ) -> np.ndarray:
        """
        Detects inter-labial mouth openness (speech articulation) in the original feed
        and composites authentic speech cavity motion onto the swapped face.
        """
        # Mouth region bounding box
        lm_left = lms[3]
        lm_right = lms[4]
        mouth_center_x = int((lm_left[0] + lm_right[0]) / 2.0)
        mouth_center_y = int((lm_left[1] + lm_right[1]) / 2.0)
        mouth_w = int(abs(lm_right[0] - lm_left[0]) * 1.3)
        mouth_h = int(28 * scale)

        x1 = max(0, mouth_center_x - mouth_w // 2)
        x2 = min(orig.shape[1], mouth_center_x + mouth_w // 2)
        y1 = max(0, mouth_center_y - mouth_h // 2)
        y2 = min(orig.shape[0], mouth_center_y + mouth_h // 2)

        if x2 <= x1 or y2 <= y1:
            return swapped

        orig_mouth = orig[y1:y2, x1:x2]
        gray_mouth = cv2.cvtColor(orig_mouth, cv2.COLOR_BGR2GRAY)

        # Measure vertical gradient energy (open cavity vs closed lips)
        sob_y = cv2.Sobel(gray_mouth, cv2.CV_32F, 0, 1, ksize=3)
        aperture_energy = float(np.mean(np.abs(sob_y))) / 255.0

        # EMA smooth aperture with first-frame bootstrap
        prev = state.get("speech_aperture")
        if prev is None:
            cur_aperture = aperture_energy
        else:
            cur_aperture = self.smoothing_alpha * prev + (1.0 - self.smoothing_alpha) * aperture_energy
        state["speech_aperture"] = cur_aperture

        # If speech aperture exceeds threshold, blend authentic oral cavity articulation
        if cur_aperture > 0.02:
            # Build soft elliptical articulation mask
            rh, rw = y2 - y1, x2 - x1
            mask = np.zeros((rh, rw), dtype=np.float32)
            cv2.ellipse(
                mask,
                (rw // 2, rh // 2),
                (int(rw * 0.42), int(rh * 0.38)),
                0,
                0,
                360,
                1.0,
                -1,
            )
            mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=rw * 0.12, sigmaY=rh * 0.12)
            alpha = np.clip(mask * strength * min(1.0, cur_aperture * 4.0), 0.0, 1.0)[:, :, np.newaxis]

            swapped_patch = swapped[y1:y2, x1:x2].astype(np.float32)
            orig_patch = orig_mouth.astype(np.float32)

            blended_patch = swapped_patch * (1.0 - alpha) + orig_patch * alpha
            swapped[y1:y2, x1:x2] = np.clip(blended_patch, 0, 255).astype(np.uint8)

        return swapped

    def _transfer_smile_energy(
        self,
        orig: np.ndarray,
        swapped: np.ndarray,
        lms: np.ndarray,
        state: Dict[str, Any],
        scale: float,
        strength: float,
    ) -> np.ndarray:
        """
        Extracts smile curvature / bilateral lip corner elevation from original feed
        and harmonizes cheek micro-tensions on the synthetic face.
        """
        lm_left = lms[3]
        lm_right = lms[4]
        nose = lms[2]

        # Calculate mouth corner slope / elevation relative to nose-mouth axis
        corner_avg_y = (lm_left[1] + lm_right[1]) / 2.0
        nose_y = nose[1]
        dist_y = max(1.0, corner_avg_y - nose_y)

        # Relative corner elevation
        raw_smile = float(np.clip(1.0 - (dist_y / (24.0 * scale)), 0.0, 1.0))
        prev_smile = state.get("smile_energy")
        if prev_smile is None:
            cur_smile = raw_smile
        else:
            cur_smile = self.smoothing_alpha * prev_smile + (1.0 - self.smoothing_alpha) * raw_smile
        state["smile_energy"] = cur_smile

        if cur_smile > 0.15:
            # Subtle smile crease / cheek tension injection
            h, w = swapped.shape[:2]
            cx, cy = int((lm_left[0] + lm_right[0]) / 2.0), int(corner_avg_y)
            rx, ry = int(26 * scale), int(16 * scale)

            x1, x2 = max(0, cx - rx), min(w, cx + rx)
            y1, y2 = max(0, cy - ry), min(h, cy + ry)

            if x2 > x1 and y2 > y1:
                orig_cheek = orig[y1:y2, x1:x2].astype(np.float32)
                swap_cheek = swapped[y1:y2, x1:x2].astype(np.float32)

                # High frequency skin crease transfer
                orig_hp = orig_cheek - cv2.GaussianBlur(orig_cheek, (0, 0), sigmaX=3.0)
                boost = np.clip(orig_hp * (cur_smile * strength * 0.35), -20, 20)

                swapped[y1:y2, x1:x2] = np.clip(swap_cheek + boost, 0, 255).astype(np.uint8)

        return swapped

    def _transfer_brow_dynamics(
        self,
        orig: np.ndarray,
        swapped: np.ndarray,
        lms: np.ndarray,
        state: Dict[str, float],
        scale: float,
        strength: float,
    ) -> np.ndarray:
        """
        Transfers authentic eyebrow furrowing and expressivity details.
        """
        eye_left = lms[0]
        eye_right = lms[1]
        eye_y = (eye_left[1] + eye_right[1]) / 2.0

        # Eyebrow band region
        brow_y = int(eye_y - 18 * scale)
        brow_h = int(14 * scale)
        brow_w = int(abs(eye_right[0] - eye_left[0]) * 1.4)
        brow_cx = int((eye_left[0] + eye_right[0]) / 2.0)

        h, w = swapped.shape[:2]
        x1 = max(0, brow_cx - brow_w // 2)
        x2 = min(w, brow_cx + brow_w // 2)
        y1 = max(0, brow_y - brow_h // 2)
        y2 = min(h, brow_y + brow_h // 2)

        if x2 <= x1 or y2 <= y1:
            return swapped

        orig_brow = orig[y1:y2, x1:x2].astype(np.float32)
        swap_brow = swapped[y1:y2, x1:x2].astype(np.float32)

        # Extract eyebrow texture displacement
        orig_texture = orig_brow - cv2.GaussianBlur(orig_brow, (0, 0), sigmaX=2.0)
        brow_boost = np.clip(orig_texture * (strength * 0.28), -15, 15)

        swapped[y1:y2, x1:x2] = np.clip(swap_brow + brow_boost, 0, 255).astype(np.uint8)
        return swapped
