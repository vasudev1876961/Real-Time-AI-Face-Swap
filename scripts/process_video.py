"""
Command Line Tool for Offline Video and Image Face Swapping.
Usage:
    python scripts/process_video.py --input input.mp4 --output output.mp4 --target prabhas --enhance 0.5
    python scripts/process_video.py --image photo.jpg --output swapped.jpg --target rekha
"""

import os
import sys
import argparse

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.pipeline.video_processor import VideoFileProcessor
from src.core.config_loader import load_all_configs
from src.pipeline.realtime_pipeline import RealTimePipeline


def main():
    parser = argparse.ArgumentParser(description="Offline Face Swap Video & Image Processor")
    parser.add_argument("--input", "-i", type=str, help="Input video file path (.mp4, .avi, .mov)")
    parser.add_argument("--output", "-o", type=str, help="Output video file path")
    parser.add_argument("--image", type=str, help="Input portrait image path (.jpg, .png)")
    parser.add_argument("--dir", type=str, help="Input directory of images for batch processing")
    parser.add_argument("--dir-out", type=str, help="Output directory for batch images")
    parser.add_argument("--target", "-t", type=str, default="prabhas", help="Target identity ID (e.g., prabhas, chiranjeevi)")
    parser.add_argument("--enhance", type=float, default=0.60, help="Face enhancement strength [0.0, 1.0]")
    parser.add_argument("--enhancer-mode", type=str, default="onnx", choices=["onnx", "adaptive", "off"], help="Enhancer engine")
    parser.add_argument("--skin-texture", type=float, default=0.35, help="High-frequency skin pore detail transfer [0.0, 1.0]")
    parser.add_argument("--blend-method", type=str, default="multiband", choices=["multiband", "alpha", "seamless_clone"], help="Blending compositing engine")
    parser.add_argument("--mask-type", type=str, default=None, help="Mask type override (smooth_hull, distance_transform, pose_adaptive)")
    parser.add_argument("--face-mode", type=str, default="primary", choices=["primary", "all", "mapped"], help="Face swapping mode: primary (largest), all, or mapped")
    parser.add_argument("--target-map", type=str, default=None, help="Target map per track ID e.g. '1:prabhas,2:chiranjeevi'")
    parser.add_argument("--no-occlusion", action="store_true", help="Disable occlusion-aware foreground masking")
    parser.add_argument("--no-stabilize", action="store_true", help="Disable temporal motion and anti-jitter stabilization")
    # Phase 9 CLI options
    parser.add_argument("--no-eye-realism", action="store_true", help="Disable natural eye blink synchronization and catchlight injection")
    parser.add_argument("--eye-strength", type=float, default=0.70, help="Eye realism and catchlight power [0.0, 1.0]")
    parser.add_argument("--no-pose-adaptation", action="store_true", help="Disable 3D pose-adaptive boundary clamping and profile falloff")
    parser.add_argument("--specular-strength", type=float, default=0.50, help="Specular-ambient lighting transfer strength [0.0, 1.0]")

    args = parser.parse_args()

    if not args.input and not args.image and not args.dir:
        print("[ERROR] Please provide either --input (video), --image (photo), or --dir (batch folder).")
        parser.print_help()
        sys.exit(1)

    # Parse target map if provided
    parsed_target_map = None
    if args.target_map:
        parsed_target_map = {}
        for pair in args.target_map.split(","):
            if ":" in pair:
                t_id, p_id = pair.split(":", 1)
                try:
                    parsed_target_map[int(t_id.strip())] = p_id.strip()
                except ValueError:
                    pass

    print("=" * 65)
    print("Initializing Face Swap Offline Processor (Phase 9)...")
    print("=" * 65)

    a_cfg, m_cfg, t_cfg = load_all_configs()
    if args.mask_type:
        a_cfg.processing.mask_type = args.mask_type
    a_cfg.processing.enhancement_strength = args.enhance
    a_cfg.processing.enhancement_mode = args.enhancer_mode
    a_cfg.processing.texture_detail_transfer = args.skin_texture
    a_cfg.processing.blending_method = args.blend_method
    a_cfg.processing.enable_occlusion = not args.no_occlusion
    a_cfg.processing.enable_stabilization = not args.no_stabilize
    a_cfg.processing.multi_face_mode = args.face_mode

    # Phase 9 parameters
    a_cfg.processing.enable_eye_realism = not args.no_eye_realism
    a_cfg.processing.eye_realism_strength = args.eye_strength
    a_cfg.processing.enable_pose_adaptation = not args.no_pose_adaptation
    a_cfg.processing.specular_lighting_strength = args.specular_strength

    pipeline = RealTimePipeline(a_cfg, m_cfg, t_cfg)
    processor = VideoFileProcessor(pipeline)


    if args.image:
        out_img = args.output or f"outputs/swapped_{os.path.basename(args.image)}"
        print(f"Processing image: {args.image} -> {out_img} (Target: {args.target}, Mode: {args.face_mode})")
        processor.process_image(
            args.image,
            out_img,
            target_id=args.target,
            enhance_strength=args.enhance,
            face_mode=args.face_mode,
            target_map=parsed_target_map,
        )
        print(f"[SUCCESS] Saved swapped image to: {out_img}")

    elif args.dir:
        out_dir = args.dir_out or "outputs/batch_swapped"
        print(f"Batch processing images: {args.dir} -> {out_dir} (Target: {args.target}, Mode: {args.face_mode})")
        results = processor.process_directory(
            args.dir,
            out_dir,
            target_id=args.target,
            enhance_strength=args.enhance,
            face_mode=args.face_mode,
            target_map=parsed_target_map,
        )
        print(f"[SUCCESS] Processed {len(results)} images into: {out_dir}")

    elif args.input:
        out_vid = args.output or f"outputs/recordings/swapped_{os.path.basename(args.input)}"
        print(f"Processing video: {args.input} -> {out_vid} (Target: {args.target}, Mode: {args.face_mode})")

        def on_progress(curr, total, fps, eta):
            pct = (curr / max(1, total)) * 100
            print(f"\r  Progress: [{curr}/{total}] {pct:.1f}% | {fps:.1f} FPS | ETA: {eta:.1f}s", end="", flush=True)

        summary = processor.process_video(
            args.input,
            out_vid,
            target_id=args.target,
            enhance_strength=args.enhance,
            progress_callback=on_progress,
            face_mode=args.face_mode,
            target_map=parsed_target_map,
        )
        print("\n" + "=" * 65)
        print(f"[SUCCESS] Video processing complete!")
        print(f"  Output: {summary['output_path']}")
        print(f"  Processed Frames: {summary['total_frames']} ({summary['swapped_frames']} swapped)")
        print(f"  Average FPS: {summary['fps_processed']}")
        print(f"  Audio Preserved: {summary['audio_preserved']}")
        print("=" * 65)


if __name__ == "__main__":
    main()
