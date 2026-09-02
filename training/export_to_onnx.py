"""
Export Trained PyTorch Hero Model Checkpoint to ONNX.
Produces a production-ready ONNX model compatible with the real-time camera app.
"""

import os
import sys
import argparse
import numpy as np

# Ensure project root is in sys.path when running this script directly
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

TORCH_AVAILABLE = False
ONNX_AVAILABLE = False
import_err_msg = ""

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError as e:
    import_err_msg = f"PyTorch is required ({e}). Install with: pip install torch torchvision"

try:
    import onnx
    ONNX_AVAILABLE = True
except ImportError as e:
    if not import_err_msg:
        import_err_msg = f"ONNX is required ({e}). Install with: pip install onnx"

try:
    from training.train_hero_model import FaceSwapModel
except ImportError:
    try:
        from train_hero_model import FaceSwapModel
    except ImportError as e:
        FaceSwapModel = None
        if not import_err_msg:
            import_err_msg = f"Failed to import FaceSwapModel: {e}"


def export_checkpoint_to_onnx(
    checkpoint_path: str = "checkpoints/hero_model_best.pt",
    output_onnx_path: str = "models/face_swap/model.onnx",
    input_resolution: int = 128,
):
    if not TORCH_AVAILABLE or not ONNX_AVAILABLE or FaceSwapModel is None:
        print(f"[ERROR] {import_err_msg or 'Required dependencies are missing.'}")
        return

    if not os.path.isfile(checkpoint_path):
        print(f"[ERROR] Checkpoint file not found at '{checkpoint_path}'")
        return

    os.makedirs(os.path.dirname(output_onnx_path), exist_ok=True)
    print("=" * 65)
    print("Exporting Trained PyTorch Hero Model to ONNX...")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Output ONNX: {output_onnx_path}")
    print("=" * 65)

    model = FaceSwapModel()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    # Dummy input: (1, 3, 128, 128)
    dummy_input = torch.randn(1, 3, input_resolution, input_resolution, dtype=torch.float32)

    torch.onnx.export(
        model,
        dummy_input,
        output_onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["target"],
        output_names=["output"],
        dynamic_axes={
            "target": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )

    print("\n[SUCCESS] Model successfully exported to ONNX!")
    print(f"  File: {output_onnx_path} ({round(os.path.getsize(output_onnx_path)/(1024*1024), 2)} MB)")
    print("\nYou can now launch the real-time camera app with your custom trained model:")
    print("  python scripts/run_app.py")
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export trained model checkpoint to ONNX")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/hero_model_best.pt", help="Path to .pt checkpoint")
    parser.add_argument("--output", type=str, default="models/face_swap/model.onnx", help="Destination ONNX path")
    args = parser.parse_args()

    export_checkpoint_to_onnx(
        checkpoint_path=args.checkpoint,
        output_onnx_path=args.output,
    )
