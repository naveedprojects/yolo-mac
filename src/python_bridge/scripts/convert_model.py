#!/usr/bin/env python3
"""
YOLOv12 Model Conversion Script

Converts PyTorch YOLO models to CoreML format with various quantization options.

Usage:
    python convert_model.py --input model.pt --output model.mlpackage --quantize fp16
"""

import argparse
import sys
import os
from pathlib import Path


def check_dependencies():
    """Check if required packages are installed."""
    missing = []

    try:
        import torch
    except ImportError:
        missing.append("torch")

    try:
        import coremltools
    except ImportError:
        missing.append("coremltools")

    try:
        from ultralytics import YOLO
    except ImportError:
        missing.append("ultralytics")

    if missing:
        print(f"Missing required packages: {', '.join(missing)}")
        print("Install with: pip install torch coremltools ultralytics")
        sys.exit(1)


def convert_model(args):
    """Convert model to CoreML format."""
    import torch
    import coremltools as ct
    from ultralytics import YOLO

    print(f"Loading model: {args.input}")
    model = YOLO(args.input)

    # Build export arguments
    export_args = {
        'format': 'coreml',
        'imgsz': [args.height, args.width],
        'nms': args.nms,
    }

    # Quantization options
    if args.quantize == 'fp16':
        export_args['half'] = True
        print("Using FP16 quantization")
    elif args.quantize == 'int8':
        export_args['int8'] = True
        if args.calibration_data:
            export_args['data'] = args.calibration_data
        print("Using INT8 quantization")

    print(f"Exporting to CoreML...")
    result_path = model.export(**export_args)

    # Move to output path
    import shutil
    output_path = Path(args.output)
    if output_path.exists():
        if output_path.is_dir():
            shutil.rmtree(output_path)
        else:
            output_path.unlink()

    shutil.move(result_path, args.output)

    print(f"Model saved to: {args.output}")

    # Print model info
    if output_path.is_dir():
        size = sum(f.stat().st_size for f in output_path.rglob('*') if f.is_file())
    else:
        size = output_path.stat().st_size

    print(f"Model size: {size / (1024*1024):.2f} MB")


def apply_advanced_quantization(args):
    """Apply advanced quantization using coremltools."""
    import coremltools as ct
    from coremltools.optimize.coreml import (
        OpLinearQuantizerConfig,
        OptimizationConfig,
        linear_quantize_weights
    )

    print(f"Loading CoreML model: {args.input}")
    model = ct.models.MLModel(args.input)

    # Configure quantization
    if args.quantize == 'int8':
        op_config = OpLinearQuantizerConfig(
            mode="linear_symmetric",
            dtype="int8",
            granularity="per_channel",
            weight_threshold=512
        )
    elif args.quantize == 'int4':
        op_config = OpLinearQuantizerConfig(
            mode="linear_symmetric",
            dtype="int4",
            granularity="per_block",
            block_size=32
        )
    else:
        op_config = OpLinearQuantizerConfig(dtype="float16")

    print(f"Applying {args.quantize.upper()} quantization...")
    config = OptimizationConfig(global_config=op_config)
    quantized_model = linear_quantize_weights(model, config=config)

    print(f"Saving quantized model to: {args.output}")
    quantized_model.save(args.output)

    # Print model info
    output_path = Path(args.output)
    if output_path.is_dir():
        size = sum(f.stat().st_size for f in output_path.rglob('*') if f.is_file())
    else:
        size = output_path.stat().st_size

    print(f"Quantized model size: {size / (1024*1024):.2f} MB")


def main():
    parser = argparse.ArgumentParser(
        description="Convert YOLOv12 models to CoreML format"
    )

    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input model path (.pt for PyTorch, .mlpackage for CoreML)"
    )

    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output model path (.mlpackage)"
    )

    parser.add_argument(
        "--quantize", "-q",
        choices=["none", "fp16", "int8", "int4", "w8a8"],
        default="fp16",
        help="Quantization type (default: fp16)"
    )

    parser.add_argument(
        "--width", "-W",
        type=int,
        default=640,
        help="Input width (default: 640)"
    )

    parser.add_argument(
        "--height", "-H",
        type=int,
        default=640,
        help="Input height (default: 640)"
    )

    parser.add_argument(
        "--nms",
        action="store_true",
        help="Include NMS in model"
    )

    parser.add_argument(
        "--calibration-data", "-c",
        help="Path to calibration data for INT8 quantization"
    )

    args = parser.parse_args()

    check_dependencies()

    input_path = Path(args.input)

    if input_path.suffix in ['.pt', '.pth']:
        # Convert from PyTorch
        convert_model(args)
    elif input_path.suffix == '.mlpackage' or input_path.is_dir():
        # Apply quantization to existing CoreML model
        apply_advanced_quantization(args)
    else:
        print(f"Unsupported input format: {input_path.suffix}")
        sys.exit(1)


if __name__ == "__main__":
    main()
