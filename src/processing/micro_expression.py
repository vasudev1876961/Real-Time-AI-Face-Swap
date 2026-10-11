"""
Dynamic Micro-Expression Wrinkle & Skin Strain Field Transfer Engine (Phase 16).
Transfers living facial expression dynamics, nasolabial smile folds, forehead
furrows, and periocular crow's feet from source subject to swapped face geometry.
"""

from typing import Optional, Tuple, Dict, Any
import math
import cv2
import numpy as np

from src.utils.logger import get_logger

logger = get_logger("MicroExpression")


class MicroExpressionEngine:
    """
    Overcomes the synthetic 'plastic dead skin' artifact by capturing high-frequency
    dermal strain differentials from live expressions and injecting dynamic micro-wrinkles
    into the swapped face dermal layer without distorting target identity geometry.
    """

    def __init__(
        self,
        default_strength: float = 0.60,
        default_sensitivity: float = 0.75,
        temporal_smoothing: float = 0.60,
    ):
        """
        Args:
            default_strength: Wrinkle injection intensity [0.0, 1.0].
            default_sensitivity: Strain activation trigger threshold sensitivity [0.0, 1.0].
            temporal_smoothing: Exponential moving average factor for strain energies.
        """
        self.default_strength = float(np.clip(default_strength, 0.0, 1.0))
        self.default_sensitivity = float(np.clip(default_sensitivity, 0.0, 1.0))
        self.temporal_smoothing = float(np.clip(temporal_smoothing, 0.0, 0.95))

        self._track_strains: Dict[int, Dict[str, float]] = {}
        self._last_metrics: Dict[str, Any] = {
            "smile_strain": 0.0,
            "brow_strain": 0.0,
            "crow_feet_strain": 0.0,
            "composite_strain": 0.0,
            "active_wrinkles": False,
        }

    def calculate_expression_strain(
        self,
        landmarks: Optional[np.ndarray],
        track_id: int = 1,
    ) -> Dict[str, float]:
        """
        Calculates geometric facial strain energies across three anatomical expression zones:
        1. smile_strain: Nasolabial fold activation (mouth widening & lip corner raise)
        2. brow_strain: Glabella / forehead furrows (brow compression or lift)
        3. crow_feet_strain: Periocular squinting / lateral eye compression

        Returns:
            Dict containing individual and composite strain values in range [0.0, 1.0].
        """
        if landmarks is None or len(landmarks) < 5:
            return {
                "smile_strain": 0.0,
                "brow_strain": 0.0,
                "crow_feet_strain": 0.0,
                "composite_strain": 0.0,
            }

        pts = np.asarray(landmarks, dtype=np.float32)

        # Standard 5-point layout: 0: left eye, 1: right eye, 2: nose, 3: left mouth, 4: right mouth
        if len(pts) >= 5:
            left_eye = pts[0]
            right_eye = pts[1]
            nose_tip = pts[2]
            left_mouth = pts[3]
            right_mouth = pts[4]

            inter_eye = float(np.linalg.norm(right_eye - left_eye))
            mouth_width = float(np.linalg.norm(right_mouth - left_mouth))
            scale = max(10.0, inter_eye)

            # 1. Smile strain: Mouth widening ratio and upward mouth curve
            # Baseline neutral ratio ~ 0.88 - 0.92
            mouth_ratio = mouth_width / scale
            mouth_mid_y = (left_mouth[1] + right_mouth[1]) / 2.0
            nose_mouth_dist = abs(mouth_mid_y - nose_tip[1]) / scale

            # Smile activates when mouth widens or corners elevate
            raw_smile = (mouth_ratio - 0.88) / 0.35
            smile_strain = float(np.clip(raw_smile, 0.0, 1.0))

            # 2. Brow / Forehead strain
            # For 5 points: evaluate vertical distance from eyes to nose
            eye_mid_y = (left_eye[1] + right_eye[1]) / 2.0
            eye_nose_dist = abs(nose_tip[1] - eye_mid_y) / scale
            # Relative deformation from standard 0.58 baseline
            raw_brow = abs(eye_nose_dist - 0.58) / 0.22
            brow_strain = float(np.clip(raw_brow, 0.0, 1.0))

            # 3. Periocular / Crow's feet strain
            # Squinting naturally accompanies hearty smiles or intense expressions
            raw_crow = (smile_strain * 0.75) + (brow_strain * 0.25)
            crow_strain = float(np.clip(raw_crow, 0.0, 1.0))
        else:
            smile_strain, brow_strain, crow_strain = 0.0, 0.0, 0.0

        # Composite strain
        composite = max(smile_strain, brow_strain, crow_strain)

        # Apply temporal smoothing
        prev = self._track_strains.get(track_id)
        if prev is not None and self.temporal_smoothing > 0.01:
            alpha = 1.0 - self.temporal_smoothing
            smile_strain = alpha * smile_strain + self.temporal_smoothing * prev["smile_strain"]
            brow_strain = alpha * brow_strain + self.temporal_smoothing * prev["brow_strain"]
            crow_strain = alpha * crow_strain + self.temporal_smoothing * prev["crow_feet_strain"]
            composite = alpha * composite + self.temporal_smoothing * prev["composite_strain"]

        result = {
            "smile_strain": float(np.clip(smile_strain, 0.0, 1.0)),
            "brow_strain": float(np.clip(brow_strain, 0.0, 1.0)),
            "crow_feet_strain": float(np.clip(crow_strain, 0.0, 1.0)),
            "composite_strain": float(np.clip(composite, 0.0, 1.0)),
        }
        self._track_strains[track_id] = result
        return result

    def extract_high_frequency_wrinkles(self, original_crop: np.ndarray) -> np.ndarray:
        """
        Extracts high-frequency cutaneous skin detail using Difference-of-Gaussians (DoG)
        on the original facial crop luminance. Rejects low-frequency macro-lighting.

        Returns:
            detail_map: float32 array in range [-1.0, 1.0].
        """
        if original_crop is None or original_crop.size == 0:
            return np.zeros((128, 128), dtype=np.float32)

        gray = cv2.cvtColor(original_crop, cv2.COLOR_BGR2GRAY).astype(np.float32)

        # DoG: Fine skin folds (sigma=1.0) vs Shading envelope (sigma=3.2)
        fine_layer = cv2.GaussianBlur(gray, (0, 0), sigmaX=1.0, sigmaY=1.0)
        shading_layer = cv2.GaussianBlur(gray, (0, 0), sigmaX=3.2, sigmaY=3.2)

        dog = (fine_layer - shading_layer) / 128.0
        return np.clip(dog, -1.0, 1.0).astype(np.float32)

    def build_anatomical_zone_masks(
        self,
        crop_shape: Tuple[int, int] = (128, 128),
        landmarks: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Synthesizes anatomically-grounded spatial masks for expression wrinkles:
        - nasolabial: Bilateral elliptical zones extending from nasal wings past mouth corners
        - forehead: Transverse band across the frontal bone / glabella
        - crow_feet: Lateral temporal wedges beside both eye corners
        """
        h, w = crop_shape[:2]
        nasolabial_mask = np.zeros((h, w), dtype=np.float32)
        forehead_mask = np.zeros((h, w), dtype=np.float32)
        crow_feet_mask = np.zeros((h, w), dtype=np.float32)

        # Standard landmark positions (approximate 128x128 space)
        cx, cy = w * 0.5, h * 0.5

        # 1. Nasolabial folds (Bilateral ellipses flanking nose to mouth)
        cv2.ellipse(nasolabial_mask, (int(w * 0.35), int(h * 0.65)), (int(w * 0.10), int(h * 0.18)), 25, 0, 360, 1.0, -1)
        cv2.ellipse(nasolabial_mask, (int(w * 0.65), int(h * 0.65)), (int(w * 0.10), int(h * 0.18)), -25, 0, 360, 1.0, -1)

        # 2. Forehead & Glabella
        cv2.ellipse(forehead_mask, (int(cx), int(h * 0.22)), (int(w * 0.32), int(h * 0.12)), 0, 0, 360, 1.0, -1)

        # 3. Crow's feet (Lateral eye zones)
        cv2.ellipse(crow_feet_mask, (int(w * 0.18), int(h * 0.42)), (int(w * 0.08), int(h * 0.10)), 0, 0, 360, 1.0, -1)
        cv2.ellipse(crow_feet_mask, (int(w * 0.82), int(h * 0.42)), (int(w * 0.08), int(h * 0.10)), 0, 0, 360, 1.0, -1)

        # Apply smooth Gaussian falloff to avoid harsh transitions
        ksize = max(5, int(min(h, w) * 0.12) | 1)
        nasolabial_mask = cv2.GaussianBlur(nasolabial_mask, (ksize, ksize), 3.0)
        forehead_mask = cv2.GaussianBlur(forehead_mask, (ksize, ksize), 3.0)
        crow_feet_mask = cv2.GaussianBlur(crow_feet_mask, (ksize, ksize), 3.0)

        return {
            "nasolabial": np.clip(nasolabial_mask, 0.0, 1.0),
            "forehead": np.clip(forehead_mask, 0.0, 1.0),
            "crow_feet": np.clip(crow_feet_mask, 0.0, 1.0),
        }

    def transfer_micro_expressions(
        self,
        original_crop: np.ndarray,
        swapped_crop: np.ndarray,
        landmarks: Optional[np.ndarray] = None,
        strength: Optional[float] = None,
        sensitivity: Optional[float] = None,
        track_id: int = 1,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Extracts live expression strain and transplants micro-wrinkles into swapped face.

        Args:
            original_crop: Aligned live camera face crop (H, W, 3).
            swapped_crop: Neural swapped face crop (H, W, 3).
            landmarks: Optional face landmarks.
            strength: Wrinkle transfer intensity [0.0, 1.0].
            sensitivity: Trigger threshold sensitivity [0.0, 1.0].
            track_id: Face tracking ID.

        Returns:
            Tuple[enhanced_crop, telemetry_metrics].
        """
        if original_crop is None or swapped_crop is None:
            return swapped_crop, self._last_metrics

        str_val = self.default_strength if strength is None else float(np.clip(strength, 0.0, 1.0))
        sens_val = self.default_sensitivity if sensitivity is None else float(np.clip(sensitivity, 0.0, 1.0))

        if str_val < 0.02:
            return swapped_crop.copy(), self._last_metrics

        h, w = swapped_crop.shape[:2]
        if original_crop.shape[:2] != (h, w):
            original_crop = cv2.resize(original_crop, (w, h), interpolation=cv2.INTER_LINEAR)

        # 1. Compute dynamic strain
        strains = self.calculate_expression_strain(landmarks, track_id=track_id)
        comp_strain = strains["composite_strain"]

        # 2. Sensitivity-modulated activation threshold
        activation_thresh = max(0.05, 0.35 * (1.0 - sens_val))
        if comp_strain < activation_thresh:
            self._last_metrics = {
                **strains,
                "active_wrinkles": False,
            }
            return swapped_crop.copy(), self._last_metrics

        # 3. High-frequency cutaneous wrinkle detail from original face
        wrinkle_detail = self.extract_high_frequency_wrinkles(original_crop)

        # 4. Spatial expression zone weighting
        zones = self.build_anatomical_zone_masks(crop_shape=(h, w), landmarks=landmarks)
        strain_weight_field = (
            strains["smile_strain"] * zones["nasolabial"]
            + strains["brow_strain"] * zones["forehead"]
            + strains["crow_feet_strain"] * zones["crow_feet"]
        )
        strain_weight_field = np.clip(strain_weight_field, 0.0, 1.0)

        # 5. Modulate wrinkle delta
        effective_scale = str_val * sens_val * 48.0  # Luminance units
        wrinkle_delta = (wrinkle_detail * strain_weight_field * effective_scale).astype(np.float32)

        # 6. Inject into swapped face luminance in YCrCb color space
        ycrcb = cv2.cvtColor(swapped_crop, cv2.COLOR_BGR2YCrCb).astype(np.float32)
        y_channel, cr_channel, cb_channel = cv2.split(ycrcb)

        # Add strain wrinkles strictly to Y (luminance), keeping skin chrominance pristine
        y_modified = np.clip(y_channel + wrinkle_delta, 0.0, 255.0)

        merged = cv2.merge([y_modified, cr_channel, cb_channel])
        result = cv2.cvtColor(merged.astype(np.uint8), cv2.COLOR_YCrCb2BGR)

        self._last_metrics = {
            "smile_strain": round(strains["smile_strain"], 3),
            "brow_strain": round(strains["brow_strain"], 3),
            "crow_feet_strain": round(strains["crow_feet_strain"], 3),
            "composite_strain": round(comp_strain, 3),
            "active_wrinkles": True,
        }

        return result, self._last_metrics

    def get_telemetry(self) -> Dict[str, Any]:
        """Returns consolidated micro-expression telemetry."""
        return dict(self._last_metrics)
