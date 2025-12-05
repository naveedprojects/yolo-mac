#!/usr/bin/env python3
"""
YOLOv8/v11/v12 ANE Benchmark Script

Downloads all model variants, converts to CoreML, and benchmarks FPS on video.
"""

import os
import sys
import time
import json
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Check for ultralytics
try:
    from ultralytics import YOLO
    import coremltools as ct
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install ultralytics coremltools opencv-python")
    sys.exit(1)

# Configuration
BENCHMARK_DIR = Path(__file__).parent
MODELS_DIR = BENCHMARK_DIR / "models"
RESULTS_DIR = BENCHMARK_DIR / "results"
VIDEO_PATH = BENCHMARK_DIR / "video" / "test_video.mp4"

# Model variants to test
MODEL_VARIANTS = ['n', 's', 'm', 'l', 'x']  # nano, small, medium, large, extra-large

# Model families
MODEL_FAMILIES = {
    'yolov8': 'yolov8',
    'yolov11': 'yolo11',
    'yolov12': 'yolov12',
}

# Benchmark settings
WARMUP_FRAMES = 10
BENCHMARK_FRAMES = 100  # Number of frames to benchmark (None = all frames)
INPUT_SIZE = 640


def setup_directories():
    """Create necessary directories."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for family in MODEL_FAMILIES.keys():
        (MODELS_DIR / family).mkdir(exist_ok=True)


def download_model(family: str, variant: str) -> Path:
    """Download a YOLO model."""
    model_name = f"{MODEL_FAMILIES[family]}{variant}"
    pt_path = MODELS_DIR / family / f"{model_name}.pt"

    if pt_path.exists():
        print(f"  [SKIP] {model_name}.pt already exists")
        return pt_path

    print(f"  [DOWNLOAD] {model_name}...")
    try:
        model = YOLO(f"{model_name}.pt")
        # Move to our directory
        src_path = Path(f"{model_name}.pt")
        if src_path.exists():
            src_path.rename(pt_path)
        return pt_path
    except Exception as e:
        print(f"  [ERROR] Failed to download {model_name}: {e}")
        return None


def convert_to_coreml(pt_path: Path, family: str, variant: str) -> Path:
    """Convert PyTorch model to CoreML."""
    model_name = f"{MODEL_FAMILIES[family]}{variant}"
    mlpackage_path = MODELS_DIR / family / f"{model_name}.mlpackage"

    if mlpackage_path.exists():
        print(f"  [SKIP] {model_name}.mlpackage already exists")
        return mlpackage_path

    print(f"  [CONVERT] {model_name} to CoreML...")
    try:
        model = YOLO(str(pt_path))
        model.export(
            format='coreml',
            imgsz=INPUT_SIZE,
            half=True,  # FP16 for ANE
            nms=False,
        )

        # Find and move the exported model
        exported = Path(f"{model_name}.mlpackage")
        if exported.exists():
            import shutil
            if mlpackage_path.exists():
                shutil.rmtree(mlpackage_path)
            shutil.move(str(exported), str(mlpackage_path))

        return mlpackage_path
    except Exception as e:
        print(f"  [ERROR] Failed to convert {model_name}: {e}")
        return None


def benchmark_model_on_video(model_path: Path, video_path: Path, warmup: int = 10, max_frames: int = None) -> dict:
    """Benchmark a model on video frames."""
    results = {
        'model': model_path.stem,
        'path': str(model_path),
        'frames_processed': 0,
        'total_time_ms': 0,
        'inference_times_ms': [],
        'fps': 0,
        'avg_inference_ms': 0,
        'min_inference_ms': 0,
        'max_inference_ms': 0,
        'error': None,
    }

    try:
        # Load model
        model = YOLO(str(model_path))

        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            results['error'] = "Failed to open video"
            return results

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = cap.get(cv2.CAP_PROP_FPS)

        print(f"    Video: {total_frames} frames @ {video_fps:.1f} FPS")

        # Determine frames to process
        if max_frames:
            frames_to_process = min(max_frames + warmup, total_frames)
        else:
            frames_to_process = total_frames

        inference_times = []
        frame_count = 0

        while frame_count < frames_to_process:
            ret, frame = cap.read()
            if not ret:
                break

            # Run inference
            start = time.perf_counter()
            _ = model(frame, imgsz=INPUT_SIZE, verbose=False)
            end = time.perf_counter()

            inference_time_ms = (end - start) * 1000

            # Skip warmup frames
            if frame_count >= warmup:
                inference_times.append(inference_time_ms)

            frame_count += 1

            # Progress update
            if frame_count % 50 == 0:
                print(f"    Processed {frame_count}/{frames_to_process} frames...")

        cap.release()

        # Calculate statistics
        if inference_times:
            results['frames_processed'] = len(inference_times)
            results['inference_times_ms'] = inference_times
            results['total_time_ms'] = sum(inference_times)
            results['avg_inference_ms'] = np.mean(inference_times)
            results['min_inference_ms'] = np.min(inference_times)
            results['max_inference_ms'] = np.max(inference_times)
            results['fps'] = 1000 / results['avg_inference_ms']
            results['p50_ms'] = np.percentile(inference_times, 50)
            results['p95_ms'] = np.percentile(inference_times, 95)
            results['p99_ms'] = np.percentile(inference_times, 99)

    except Exception as e:
        results['error'] = str(e)

    return results


def run_benchmarks():
    """Run all benchmarks."""
    setup_directories()

    if not VIDEO_PATH.exists():
        print(f"ERROR: Video not found at {VIDEO_PATH}")
        return

    all_results = {
        'timestamp': datetime.now().isoformat(),
        'video': str(VIDEO_PATH),
        'input_size': INPUT_SIZE,
        'warmup_frames': WARMUP_FRAMES,
        'benchmark_frames': BENCHMARK_FRAMES,
        'models': {},
    }

    # Get system info
    import platform
    all_results['system'] = {
        'platform': platform.platform(),
        'processor': platform.processor(),
        'python': platform.python_version(),
    }

    print("=" * 60)
    print("YOLO ANE Benchmark")
    print("=" * 60)
    print(f"Video: {VIDEO_PATH}")
    print(f"Input size: {INPUT_SIZE}x{INPUT_SIZE}")
    print(f"Warmup: {WARMUP_FRAMES} frames")
    print(f"Benchmark: {BENCHMARK_FRAMES} frames")
    print("=" * 60)

    # Process each model family and variant
    for family in MODEL_FAMILIES.keys():
        print(f"\n[{family.upper()}]")
        all_results['models'][family] = {}

        for variant in MODEL_VARIANTS:
            model_name = f"{MODEL_FAMILIES[family]}{variant}"
            print(f"\n  Processing {model_name}...")

            # Download model
            pt_path = download_model(family, variant)
            if not pt_path or not pt_path.exists():
                print(f"  [SKIP] Could not get {model_name}")
                continue

            # Convert to CoreML
            mlpackage_path = convert_to_coreml(pt_path, family, variant)
            if not mlpackage_path or not mlpackage_path.exists():
                print(f"  [SKIP] Could not convert {model_name}")
                continue

            # Benchmark
            print(f"  [BENCHMARK] {model_name}...")
            results = benchmark_model_on_video(
                mlpackage_path,
                VIDEO_PATH,
                warmup=WARMUP_FRAMES,
                max_frames=BENCHMARK_FRAMES
            )

            if results['error']:
                print(f"  [ERROR] {results['error']}")
            else:
                print(f"  [RESULT] {results['fps']:.1f} FPS (avg: {results['avg_inference_ms']:.1f}ms)")

            # Store results (without raw inference times to save space)
            results_summary = {k: v for k, v in results.items() if k != 'inference_times_ms'}
            all_results['models'][family][variant] = results_summary

    # Save results
    results_file = RESULTS_DIR / f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n\nResults saved to: {results_file}")

    # Print summary table
    print_summary(all_results)

    return all_results


def print_summary(results: dict):
    """Print a summary table of results."""
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 80)
    print(f"{'Model':<20} {'FPS':>10} {'Avg (ms)':>12} {'P50 (ms)':>12} {'P95 (ms)':>12}")
    print("-" * 80)

    all_fps = []

    for family in ['yolov8', 'yolov11', 'yolov12']:
        if family not in results['models']:
            continue

        for variant in MODEL_VARIANTS:
            if variant not in results['models'][family]:
                continue

            r = results['models'][family][variant]
            model_name = f"{MODEL_FAMILIES[family]}{variant}"

            if r.get('error'):
                print(f"{model_name:<20} {'ERROR':>10}")
            else:
                fps = r.get('fps', 0)
                avg = r.get('avg_inference_ms', 0)
                p50 = r.get('p50_ms', 0)
                p95 = r.get('p95_ms', 0)
                print(f"{model_name:<20} {fps:>10.1f} {avg:>12.1f} {p50:>12.1f} {p95:>12.1f}")
                all_fps.append((model_name, fps))

        print("-" * 80)

    # Print fastest models
    if all_fps:
        all_fps.sort(key=lambda x: x[1], reverse=True)
        print("\nFastest models:")
        for i, (name, fps) in enumerate(all_fps[:5], 1):
            print(f"  {i}. {name}: {fps:.1f} FPS")


if __name__ == "__main__":
    run_benchmarks()
