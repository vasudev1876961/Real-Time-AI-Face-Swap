"""
Interactive Camera Diagnostics and Benchmark Tool.
Probes connected DirectShow/MSMF/OpenCV capture devices and tests frame rate throughput.
"""

import argparse
import sys
import time
import cv2

from src.camera.camera_manager import CameraManager


def probe_and_test_camera(device_index: int = 0, num_frames: int = 60) -> bool:
    print(f"\n==========================================")
    print(f" Camera Probe & Throughput Diagnostic")
    print(f"==========================================\n")

    print("[*] Discovering connected video devices...")
    devices = CameraManager.discover_cameras(max_probe=4)
    if not devices:
        print("[-] No camera devices discovered!")
    else:
        for dev in devices:
            print(f"  - Device [{dev.index}]: {dev.name} (Active: {dev.is_available})")

    print(f"\n[*] Probing capture on camera index: {device_index}...")
    cap = cv2.VideoCapture(device_index, cv2.CAP_ANY)
    if not cap.isOpened():
        print(f"[-] Could not open camera device {device_index}.")
        return False

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap_fps = cap.get(cv2.CAP_PROP_FPS)
    backend_name = cap.getBackendName()

    print(f"[+] Camera opened successfully!")
    print(f"    Backend:    {backend_name}")
    print(f"    Resolution: {width} x {height}")
    print(f"    Reported FPS: {cap_fps}")

    print(f"\n[*] Benchmarking {num_frames} frames acquisition throughput...")
    t0 = time.perf_counter()
    captured = 0

    for i in range(num_frames):
        ret, frame = cap.read()
        if not ret or frame is None:
            print(f"[-] Frame {i} read failed.")
            break
        captured += 1

    elapsed = time.perf_counter() - t0
    cap.release()

    if captured > 0 and elapsed > 0:
        actual_fps = captured / elapsed
        print(f"[+] Benchmark Complete!")
        print(f"    Captured:   {captured} frames in {elapsed:.2f} seconds")
        print(f"    Actual FPS: {actual_fps:.2f} FPS")
        print(f"    Frame Time: {(elapsed / captured) * 1000.0:.2f} ms/frame")
        return True
    else:
        print("[-] Benchmark failed to capture any frames.")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-Time AI Face Swap - Camera Diagnostic")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index (default: 0)")
    parser.add_argument("--frames", type=int, default=30, help="Number of benchmark frames (default: 30)")
    args = parser.parse_args()

    success = probe_and_test_camera(args.camera, args.frames)
    sys.exit(0 if success else 1)
