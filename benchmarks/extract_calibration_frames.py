#!/usr/bin/env python3
"""
Extract frames from video for INT8 calibration.
"""

import cv2
import os
from pathlib import Path

VIDEO_PATH = Path(__file__).parent / "video" / "test_video.mp4"
OUTPUT_DIR = Path(__file__).parent / "calibration_images"
NUM_FRAMES = 100

def extract_frames():
    """Extract evenly spaced frames from video."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        print(f"ERROR: Cannot open video {VIDEO_PATH}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video has {total_frames} frames")

    # Calculate interval to get evenly spaced frames
    interval = total_frames // NUM_FRAMES
    print(f"Extracting {NUM_FRAMES} frames (every {interval}th frame)")

    extracted = 0
    frame_idx = 0

    while extracted < NUM_FRAMES:
        # Seek to frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        if not ret:
            break

        # Save frame
        output_path = OUTPUT_DIR / f"frame_{extracted:04d}.jpg"
        cv2.imwrite(str(output_path), frame)

        extracted += 1
        frame_idx += interval

        if extracted % 20 == 0:
            print(f"  Extracted {extracted}/{NUM_FRAMES} frames...")

    cap.release()
    print(f"Done! Extracted {extracted} frames to {OUTPUT_DIR}")

if __name__ == "__main__":
    extract_frames()
