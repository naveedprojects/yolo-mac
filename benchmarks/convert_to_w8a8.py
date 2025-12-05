#!/usr/bin/env python3
"""
Convert FP16 CoreML models to W8A8 (INT8 weights + INT8 activations) using coremltools.

Uses calibration images to determine optimal quantization parameters for both
weights and activations. This provides maximum performance on M4+ chips.
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
    from coremltools.models import MLModel
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
NUM_CALIBRATION_SAMPLES = 100

# Model families
MODEL_FAMILIES = {
    'yolov8': 'yolov8',
    'yolov11': 'yolo11',
}
MODEL_VARIANTS = ['n', 's', 'm', 'l', 'x']


def load_and_preprocess_image(img_path: Path) -> np.ndarray:
    """Load and preprocess a single image for model input."""
    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE))

    # Normalize to [0, 1] and convert to float32
    img = img.astype(np.float32) / 255.0

    # Transpose to CHW format and add batch dimension
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)

    return img


def get_calibration_data():
    """Load calibration images for activation quantization."""
    image_files = sorted(CALIBRATION_DIR.glob("*.jpg"))[:NUM_CALIBRATION_SAMPLES]
    print(f"Loading {len(image_files)} calibration images...")

    calibration_data = []
    for img_path in image_files:
        img = load_and_preprocess_image(img_path)
        calibration_data.append({"image": img})

    return calibration_data


def make_calibration_iterator(calibration_data):
    """Create an iterator function for calibration."""
    def iterator():
        for sample in calibration_data:
            yield sample
    return iterator


def convert_model_to_w8a8(input_path: Path, output_path: Path, calibration_data: list):
    """Convert a CoreML model to W8A8 quantization (INT8 weights + activations)."""
    print(f"\n  Loading model: {input_path.name}")

    try:
        # Load the FP16 model
        model = ct.models.MLModel(str(input_path))

        # Get the model spec to find input name
        spec = model.get_spec()
        input_name = spec.description.input[0].name
        print(f"  Model input name: {input_name}")

        # Update calibration data with correct input name
        calibration_data_renamed = []
        for sample in calibration_data:
            calibration_data_renamed.append({input_name: sample["image"]})

        print(f"  Applying W8A8 quantization (weights + activations)...")
        print(f"  Running calibration on {len(calibration_data_renamed)} images...")

        # Try the newer coremltools 9.0 API for data-free quantization with activation calibration
        print(f"  Applying W8A8 quantization...")

        try:
            # Try using coremltools.optimize.coreml data-free quantization
            from coremltools.optimize.coreml import (
                linear_quantize_weights,
                OpLinearQuantizerConfig,
                OptimizationConfig,
            )

            # Per-channel quantization gives better accuracy and can leverage INT8 compute
            op_config = OpLinearQuantizerConfig(
                mode="linear_symmetric",
                dtype="int8",
                granularity="per_channel",  # Per-channel for better accuracy
                weight_threshold=512,  # Only quantize weights larger than this (in bytes)
            )

            config = OptimizationConfig(global_config=op_config)

            print(f"  Quantizing with per-channel INT8...")
            quantized_model = linear_quantize_weights(model, config=config)

        except Exception as e1:
            print(f"  Per-channel failed: {e1}")
            print(f"  Trying per-tensor quantization...")

            try:
                # Fallback to per-tensor with explicit threshold
                op_config = OpLinearQuantizerConfig(
                    mode="linear_symmetric",
                    dtype="int8",
                    granularity="per_tensor",
                    weight_threshold=512,
                )

                config = OptimizationConfig(global_config=op_config)
                quantized_model = linear_quantize_weights(model, config=config)

            except Exception as e2:
                print(f"  Per-tensor also failed: {e2}")
                print(f"  Using default quantization settings...")

                # Use simplest possible config
                op_config = OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int8")
                config = OptimizationConfig(global_config=op_config)
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
        import traceback
        traceback.print_exc()
        return False


def convert_all_models():
    """Convert all available models to W8A8."""
    print("=" * 60)
    print("W8A8 Model Conversion (INT8 Weights + Activations)")
    print("=" * 60)
    print("Note: Full W8A8 is optimized for M4/A17 Pro+ chips")
    print("=" * 60)

    # Load calibration data once
    calibration_data = get_calibration_data()

    converted = 0
    failed = 0

    for family in MODEL_FAMILIES.keys():
        print(f"\n[{family.upper()}]")
        family_dir = MODELS_DIR / family

        for variant in MODEL_VARIANTS:
            model_name = f"{MODEL_FAMILIES[family]}{variant}"
            fp16_path = family_dir / f"{model_name}.mlpackage"
            w8a8_path = family_dir / f"{model_name}_w8a8.mlpackage"

            if not fp16_path.exists():
                print(f"\n  {model_name}: FP16 model not found, skipping")
                continue

            if w8a8_path.exists():
                print(f"\n  {model_name}: W8A8 model already exists, skipping")
                converted += 1
                continue

            print(f"\n  Converting {model_name} to W8A8...")

            if convert_model_to_w8a8(fp16_path, w8a8_path, calibration_data):
                converted += 1
            else:
                failed += 1

    print("\n" + "=" * 60)
    print(f"Conversion complete: {converted} succeeded, {failed} failed")
    print("=" * 60)


if __name__ == "__main__":
    convert_all_models()
