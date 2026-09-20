"""
CLI Script for Multi-Portrait Identity Embedding Fusion & Celebrity Blending.
Combines multiple celebrity targets or multiple photos of a subject into a unified hybrid identity.

Usage:
    python scripts/blend_targets.py --targets prabhas,chiranjeevi --weights 0.5,0.5 --id prabhas_chiru --name "Prabhas-Chiru Hybrid"
"""

import os
import sys
import argparse
import numpy as np
import cv2

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.core.config_loader import load_all_configs
from src.targets.target_manager import get_target_manager
from src.targets.identity_fusion import IdentityEmbeddingFuser
from src.utils.logger import setup_logging, get_logger

logger = get_logger("BlendTargets")


def main():
    parser = argparse.ArgumentParser(description="Multi-Portrait Identity Embedding Fusion CLI")
    parser.add_argument("--targets", "-t", type=str, required=True, help="Comma-separated target IDs (e.g., prabhas,chiranjeevi)")
    parser.add_argument("--weights", "-w", type=str, default=None, help="Comma-separated weights (e.g., 0.5,0.5)")
    parser.add_argument("--id", type=str, required=True, help="Unique target ID for the fused identity (e.g., hybrid_prabhas_chiru)")
    parser.add_argument("--name", type=str, required=True, help="Display name for the fused identity (e.g., 'Prabhas & Chiru Hybrid')")
    parser.add_argument("--category", "-c", type=str, default="custom", help="Database category folder (default: custom)")
    args = parser.parse_args()

    setup_logging()
    app_cfg, models_cfg, targets_cfg = load_all_configs()
    tm = get_target_manager(targets_cfg)

    target_ids = [t.strip() for t in args.targets.split(",") if t.strip()]
    if len(target_ids) < 2:
        print("[ERROR] Please provide at least two target IDs to fuse.")
        sys.exit(1)

    weights = None
    if args.weights:
        weights = [float(w.strip()) for w in args.weights.split(",") if w.strip()]
        if len(weights) != len(target_ids):
            print(f"[ERROR] Weight count ({len(weights)}) does not match target count ({len(target_ids)}).")
            sys.exit(1)

    print("=" * 65)
    print(f"Blending {len(target_ids)} targets: {', '.join(target_ids)}")
    print("=" * 65)

    embeddings = []
    ref_images = []
    loaded_targets = []

    for tid in target_ids:
        t = tm.get_target_by_id(tid)
        if t is None:
            print(f"[ERROR] Target ID '{tid}' not found in database.")
            sys.exit(1)
        emb = t.embedding
        if emb is None:
            print(f"[ERROR] Target '{tid}' does not have an identity embedding.")
            sys.exit(1)
        embeddings.append(emb)
        loaded_targets.append(t)
        ref_img = cv2.imread(t.reference_image_path)
        if ref_img is not None:
            ref_images.append(ref_img)

    # Compute fused embedding
    if len(embeddings) == 2 and weights is not None:
        alpha = weights[1] / max(1e-4, (weights[0] + weights[1]))
        fused_emb = IdentityEmbeddingFuser.interpolate_identities(embeddings[0], embeddings[1], alpha=alpha)
    else:
        fused_emb = IdentityEmbeddingFuser.fuse_embeddings(embeddings, weights=weights)

    # Blend reference images for composite portrait preview
    h_ref, w_ref = 512, 512
    resized_refs = [cv2.resize(img, (w_ref, h_ref), interpolation=cv2.INTER_LANCZOS4) for img in ref_images]
    w_norm = np.array(weights if weights else [1.0] * len(resized_refs), dtype=np.float32)
    w_norm = w_norm / np.sum(w_norm)

    fused_ref = np.zeros((h_ref, w_ref, 3), dtype=np.float32)
    for img, w_val in zip(resized_refs, w_norm):
        fused_ref += img.astype(np.float32) * w_val
    fused_ref = np.clip(fused_ref, 0, 255).astype(np.uint8)

    # Persist fused target
    target = IdentityEmbeddingFuser.create_fused_target(
        target_id=args.id,
        display_name=args.name,
        category=args.category,
        fused_embedding=fused_emb,
        reference_image=fused_ref,
        source_metadata={
            "source_targets": target_ids,
            "weights": [float(w) for w in (weights or [1.0] * len(target_ids))],
        },
    )

    print(f"[SUCCESS] Fused identity '{args.name}' created successfully!")
    print(f"  Target ID: {target.target_id}")
    print(f"  Category: faces/{target.category}/{target.target_id}/")
    print(f"  Reference Image: {target.reference_image_path}")
    print(f"  Embedding Cache: faces/{target.category}/{target.target_id}/face.npy")
    print("=" * 65)


if __name__ == "__main__":
    main()
