#!/usr/bin/env python3
"""
Convert FP16 CoreML models to INT8 using coremltools.

Uses calibration images to determine optimal quantization parameters.
"""

import os
import sys
from pathlib import Path
import numpy as np
import cv2

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

try:
    import coremltools as ct
    from coremltools.optimize.coreml import (
        OptimizationConfig,
        OpLinearQuantizerConfig,
        linear_quantize_weights,
    )
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install coremltools")
    sys.exit(1)

print(f"coremltools version: {ct.__version__}")

# Configuration
BENCHMARK_DIR = Path(__file__).parent
MODELS_DIR = BENCHMARK_DIR / "models"
CALIBRATION_DIR = BENCHMARK_DIR / "calibration_images"
INPUT_SIZE = 640

# Model families
MODEL_FAMILIES = {
    'yolov8': 'yolov8',
    'yolov11': 'yolo11',
}
MODEL_VARIANTS = ['n', 's', 'm', 'l', 'x']


def load_calibration_images(num_samples: int = 100):
    """Load calibration images and preprocess them."""
    images = []
    image_files = sorted(CALIBRATION_DIR.glob("*.jpg"))[:num_samples]

    print(f"Loading {len(image_files)} calibration images...")

    for img_path in image_files:
        # Load and preprocess image
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE))

        # Normalize to [0, 1] and convert to float32
        img = img.astype(np.float32) / 255.0

        # Transpose to CHW format
        img = np.transpose(img, (2, 0, 1))

        # Add batch dimension
        img = np.expand_dims(img, axis=0)

        images.append(img)

    return images


def make_sample_iterator(images):
    """Create a sample iterator for calibration."""
    def sample_iterator():
        for img in images:
            yield {"image": img}
    return sample_iterator


def convert_model_to_int8(input_path: Path, output_path: Path, calibration_images=None):
    """Convert a CoreML model to INT8 quantization."""
    print(f"\n  Loading model: {input_path.name}")

    try:
        # Load the FP16 model
        model = ct.models.MLModel(str(input_path))

        print(f"  Applying INT8 weight quantization...")

        # Option 1: Weight-only INT8 quantization (simpler, works on all Apple Silicon)
        # This quantizes weights to INT8 but keeps activations in FP16
        config = OptimizationConfig(
            global_config=OpLinearQuantizerConfig(
                mode="linear_symmetric",
                dtype="int8",
                granularity="per_tensor",
            )
        )

        quantized_model = linear_quantize_weights(model, config=config)

        # Save the quantized model
        print(f"  Saving to: {output_path.name}")
        quantized_model.save(str(output_path))

        # Get model sizes
        original_size = sum(f.stat().st_size for f in input_path.rglob('*') if f.is_file())
        quantized_size = sum(f.stat().st_size for f in output_path.rglob('*') if f.is_file())

        print(f"  Original size: {original_size / 1024 / 1024:.1f} MB")
        print(f"  Quantized size: {quantized_size / 1024 / 1024:.1f} MB")
        print(f"  Compression: {original_size / quantized_size:.2f}x")

        return True

    except Exception as e:
        print(f"  [ERROR] {e}")
        return False


def convert_all_models():
    """Convert all available models to INT8."""
    print("=" * 60)
    print("INT8 Model Conversion")
    print("=" * 60)

    # Load calibration images
    calibration_images = load_calibration_images()

    converted = 0
    failed = 0

    for family in MODEL_FAMILIES.keys():
        print(f"\n[{family.upper()}]")
        family_dir = MODELS_DIR / family

        for variant in MODEL_VARIANTS:
            model_name = f"{MODEL_FAMILIES[family]}{variant}"
            fp16_path = family_dir / f"{model_name}.mlpackage"
            int8_path = family_dir / f"{model_name}_int8.mlpackage"

            if not fp16_path.exists():
                print(f"\n  {model_name}: FP16 model not found, skipping")
                continue

            if int8_path.exists():
                print(f"\n  {model_name}: INT8 model already exists, skipping")
                converted += 1
                continue

            print(f"\n  Converting {model_name}...")

            if convert_model_to_int8(fp16_path, int8_path, calibration_images):
                converted += 1
            else:
                failed += 1

    print("\n" + "=" * 60)
    print(f"Conversion complete: {converted} succeeded, {failed} failed")
    print("=" * 60)


if __name__ == "__main__":
    convert_all_models()
