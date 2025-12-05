# YOLO-Mac

A high-performance C++ framework for running **YOLOv8, YOLOv11, and YOLOv12** object detection models on Apple Silicon Macs. This project focuses on maximizing inference speed using the Apple Neural Engine (ANE) with comprehensive quantization support.

## Why This Project?

Apple Silicon's Neural Engine is incredibly powerful but underutilized. Our benchmarks show **ANE is 3-7x faster than GPU** for YOLO inference, yet most implementations default to GPU or CPU. This framework:

- Provides **native CoreML integration** for optimal ANE utilization
- Supports **multiple quantization levels** (FP16, INT8, W8A8) for speed/accuracy tradeoffs
- Offers a **pure C++ API** with no Python dependency for inference
- Achieves **140+ FPS** on YOLOv8n/YOLOv11n with M4 chip

## Benchmark Results (Apple M4)

We conducted comprehensive benchmarks comparing all backends and precision levels on an M4 chip with 640x640 input.

### Performance Summary (FPS)

| Model | ANE FP16 | ANE INT8 | ANE W8A8 | GPU FP16 | ANE Speedup |
|-------|----------|----------|----------|----------|-------------|
| **YOLOv8n** | 139.4 | 140.5 | 139.4 | 21.0 | **6.6x** |
| **YOLOv8s** | 116.5 | 116.0 | 114.2 | 18.9 | **6.2x** |
| **YOLOv8m** | 82.5 | 83.0 | 83.0 | 15.6 | **5.3x** |
| **YOLOv8l** | 54.9 | 57.3 | 57.7 | 14.3 | **3.8x** |
| **YOLOv8x** | 39.7 | 41.0 | 41.9 | 11.5 | **3.5x** |
| **YOLOv11n** | 139.7 | 140.6 | 138.7 | 28.4 | **4.9x** |
| **YOLOv11s** | 114.4 | 115.2 | 116.0 | 24.3 | **4.7x** |
| **YOLOv11m** | 68.1 | 77.9 | 78.7 | 19.4 | **3.5x** |
| **YOLOv11l** | 57.5 | 68.4 | 67.6 | 17.9 | **3.2x** |
| **YOLOv11x** | 35.6 | 41.2 | 41.5 | 12.4 | **2.9x** |

### Key Findings

**1. ANE Dominates GPU Performance**
- ANE provides 3-7x speedup over Metal GPU across all models
- The speedup is most dramatic for smaller models (nano/small)
- Even the largest models (YOLOv8x) see 3.5x improvement

**2. INT8 Quantization Benefits Vary by Model**
- Small models (n/s): Minimal speed difference from FP16 (~1%)
- Large models (m/l/x): Significant speedup up to **19%** (YOLOv11l)
- YOLOv11 benefits more from INT8 than YOLOv8

**3. Model Size Reduction**
- INT8/W8A8 provides consistent **~2x compression** across all models
- YOLOv8n: 6.2MB → 3.2MB
- YOLOv8x: 130.3MB → 65.3MB

**4. INT8 vs W8A8**
- **INT8**: Weight-only quantization (INT8 weights, FP16 activations) - works on all Apple Silicon
- **W8A8**: Full quantization (INT8 weights + INT8 activations) - optimized for M4/A17 Pro+
- On M4, both provide similar performance; W8A8 may show additional gains on future chips

### Latency (Milliseconds)

| Model | ANE FP16 | ANE INT8 | ANE W8A8 | GPU FP16 |
|-------|----------|----------|----------|----------|
| **YOLOv8n** | 7.2 | 7.1 | 7.2 | 47.6 |
| **YOLOv8s** | 8.6 | 8.6 | 8.8 | 52.9 |
| **YOLOv8m** | 12.1 | 12.0 | 12.0 | 64.2 |
| **YOLOv8l** | 18.2 | 17.4 | 17.3 | 69.9 |
| **YOLOv8x** | 25.2 | 24.4 | 23.9 | 87.2 |
| **YOLOv11n** | 7.2 | 7.1 | 7.2 | 35.2 |
| **YOLOv11s** | 8.7 | 8.7 | 8.6 | 41.1 |
| **YOLOv11m** | 14.7 | 12.8 | 12.7 | 51.7 |
| **YOLOv11l** | 17.4 | 14.6 | 14.8 | 55.9 |
| **YOLOv11x** | 28.1 | 24.3 | 24.1 | 81.0 |

> See [benchmarks/README.md](benchmarks/README.md) for complete methodology and detailed results.

## Supported Models

| Model Family | Variants | Status |
|--------------|----------|--------|
| **YOLOv12** | n, s, m, l, x | Ready (awaiting model release) |
| **YOLOv11** | n, s, m, l, x | Fully tested |
| **YOLOv8** | n, s, m, l, x | Fully tested |

## Features

- **Pure C++ API** - No Python required for inference
- **Multiple Backends** - CoreML (native) and ONNX Runtime with CoreML EP
- **Compute Unit Selection** - Choose between ANE, GPU, CPU, or automatic
- **Advanced Quantization** - FP16, INT8 (weight-only), W8A8 (full INT8)
- **Model Conversion Tools** - Convert PyTorch/ONNX models to optimized CoreML format
- **Comprehensive Benchmarking** - Scripts to reproduce all performance measurements

## Quick Start

### Basic Detection (C++)

```cpp
#include <yolov12/yolov12.h>

int main() {
    // Load model with ANE preference
    auto detector = yolov12::Detector::create(
        "yolov8n.mlpackage",
        yolov12::DetectorConfig::for_ane()
    );

    // Load image and detect
    auto image = yolov12::Image::from_file("test.jpg");
    auto detections = detector->detect(image);

    // Print results
    for (const auto& det : detections) {
        std::cout << det.class_name << ": "
                  << (det.confidence * 100) << "%\n";
    }
}
```

### Model Conversion & Quantization

```bash
# Convert to FP16 (recommended for M1/M2/M3)
./yolov12-convert -i yolov8n.pt -o yolov8n.mlpackage -q fp16

# Convert to INT8 with calibration data (recommended for M4+)
./yolov12-convert -i yolov8n.pt -o yolov8n_int8.mlpackage -q int8 -c ./calibration_images/

# Convert to W8A8 (full INT8, optimized for M4/A17 Pro+)
./yolov12-convert -i yolov8n.pt -o yolov8n_w8a8.mlpackage -q w8a8 -c ./calibration_images/
```

## Building

### Prerequisites

- macOS 13.0+ (Ventura or later)
- Xcode 15+ with Command Line Tools
- CMake 3.20+
- (Optional) Python 3.9+ with coremltools for quantization

### Build

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(sysctl -n hw.ncpu)
```

## Quantization Guide

### Which Quantization Should I Use?

| Your Hardware | Recommended | Why |
|---------------|-------------|-----|
| M1/M2/M3 | FP16 | Best balance of speed and accuracy |
| M4/A17 Pro+ | INT8 or W8A8 | Native INT8 compute support on ANE |
| Memory limited | INT8/W8A8 | ~2x smaller model size |
| Maximum accuracy | FP16 | No quantization error |

### Quantization Types Explained

| Type | Weights | Activations | Size Reduction | Speed Gain |
|------|---------|-------------|----------------|------------|
| **FP16** | 16-bit float | 16-bit float | Baseline | Baseline |
| **INT8** | 8-bit int | 16-bit float | ~2x smaller | 0-19% faster |
| **W8A8** | 8-bit int | 8-bit int | ~2x smaller | 0-19% faster |

**INT8** (weight-only) quantizes just the model weights to INT8 while keeping activations in FP16. This is safe and works well on all Apple Silicon.

**W8A8** quantizes both weights and activations to INT8. This requires calibration data to determine activation ranges and is optimized for M4/A17 Pro chips with enhanced INT8 compute capabilities.

## Project Structure

```
yolov12-mac/
├── include/           # C++ headers
├── src/               # C++ implementation
├── python/            # Python bindings and conversion tools
├── tools/             # CLI utilities
├── examples/          # Usage examples
├── benchmarks/        # Benchmark scripts and results
│   ├── run_benchmark.py          # ANE FP16 benchmark
│   ├── run_benchmark_int8.py     # ANE INT8 benchmark
│   ├── run_benchmark_w8a8.py     # ANE W8A8 benchmark
│   ├── run_benchmark_gpu.py      # GPU FP32 benchmark
│   ├── run_benchmark_gpu_fp16.py # GPU FP16 benchmark
│   ├── convert_to_int8.py        # INT8 conversion script
│   ├── convert_to_w8a8.py        # W8A8 conversion script
│   └── README.md                 # Detailed benchmark methodology
└── tests/             # Unit tests
```

## Running Your Own Benchmarks

See the [benchmarks/](benchmarks/) directory for scripts to reproduce our results:

```bash
cd benchmarks

# 1. Download models (requires ultralytics)
pip install ultralytics
python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

# 2. Export to CoreML
python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt').export(format='coreml')"

# 3. Run benchmarks
python3 run_benchmark.py      # ANE FP16
python3 run_benchmark_int8.py # ANE INT8 (after conversion)
```

## Recommendations by Use Case

| Use Case | Model | Precision | Expected FPS (M4) |
|----------|-------|-----------|-------------------|
| Real-time (>100 FPS) | YOLOv8n / YOLOv11n | INT8 | ~140 |
| Balanced | YOLOv8s / YOLOv11s | INT8 | ~115 |
| High accuracy | YOLOv8m / YOLOv11m | INT8 | ~80 |
| Maximum accuracy | YOLOv8l / YOLOv11l | FP16 | ~55 |
| Edge/embedded | YOLOv8n | W8A8 | ~140 |

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- [Ultralytics](https://github.com/ultralytics/ultralytics) for YOLO models
- [Apple Core ML Tools](https://github.com/apple/coremltools) for model conversion
- [ONNX Runtime](https://github.com/microsoft/onnxruntime) for ONNX support
