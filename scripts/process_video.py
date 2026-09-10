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
    parser.add_argument("--enhance", type=float, default=0.45, help="Face enhancement strength [0.0, 1.0]")
    parser.add_argument("--mask-type", type=str, default=None, help="Mask type override (smooth_hull, distance_transform, pose_adaptive)")

    args = parser.parse_args()

    if not args.input and not args.image and not args.dir:
        print("[ERROR] Please provide either --input (video), --image (photo), or --dir (batch folder).")
        parser.print_help()
        sys.exit(1)

    print("=" * 65)
    print("Initializing Face Swap Offline Processor...")
    print("=" * 65)

    a_cfg, m_cfg, t_cfg = load_all_configs()
    if args.mask_type:
        a_cfg.processing.mask_type = args.mask_type
    a_cfg.processing.enhancement_strength = args.enhance

    pipeline = RealTimePipeline(a_cfg, m_cfg, t_cfg)
    processor = VideoFileProcessor(pipeline)

    if args.image:
        out_img = args.output or f"outputs/swapped_{os.path.basename(args.image)}"
        print(f"Processing image: {args.image} -> {out_img} (Target: {args.target})")
        processor.process_image(args.image, out_img, target_id=args.target, enhance_strength=args.enhance)
        print(f"[SUCCESS] Saved swapped image to: {out_img}")

    elif args.dir:
        out_dir = args.dir_out or "outputs/batch_swapped"
        print(f"Batch processing images: {args.dir} -> {out_dir} (Target: {args.target})")
        results = processor.process_directory(args.dir, out_dir, target_id=args.target, enhance_strength=args.enhance)
        print(f"[SUCCESS] Processed {len(results)} images into: {out_dir}")

    elif args.input:
        out_vid = args.output or f"outputs/recordings/swapped_{os.path.basename(args.input)}"
        print(f"Processing video: {args.input} -> {out_vid} (Target: {args.target})")

        def on_progress(curr, total, fps, eta):
            pct = (curr / max(1, total)) * 100
            print(f"\r  Progress: [{curr}/{total}] {pct:.1f}% | {fps:.1f} FPS | ETA: {eta:.1f}s", end="", flush=True)

        summary = processor.process_video(
            args.input,
            out_vid,
            target_id=args.target,
            enhance_strength=args.enhance,
            progress_callback=on_progress,
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
