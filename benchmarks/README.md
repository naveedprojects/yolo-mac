# YOLO Benchmark Suite

Comprehensive benchmarking scripts for comparing YOLO model performance across different backends (ANE vs GPU) and precision levels (FP16, INT8, W8A8) on Apple Silicon.

## Quick Start

### 1. Install Dependencies

```bash
pip install ultralytics opencv-python coremltools numpy
```

### 2. Download Models

The benchmark scripts automatically download models, but you can also do it manually:

```bash
cd benchmarks

# Create model directories
mkdir -p models/yolov8 models/yolov11 models/yolov12

# Download and export YOLOv8 models
python3 << 'EOF'
from ultralytics import YOLO
import shutil
from pathlib import Path

for variant in ['n', 's', 'm', 'l', 'x']:
    model = YOLO(f'yolov8{variant}.pt')
    model.export(format='coreml', imgsz=640)

    # Move files to correct location
    pt_file = Path(f'yolov8{variant}.pt')
    ml_dir = Path(f'yolov8{variant}.mlpackage')

    if pt_file.exists():
        shutil.move(str(pt_file), f'models/yolov8/yolov8{variant}.pt')
    if ml_dir.exists():
        shutil.move(str(ml_dir), f'models/yolov8/yolov8{variant}.mlpackage')
EOF

# Download and export YOLOv11 models
python3 << 'EOF'
from ultralytics import YOLO
import shutil
from pathlib import Path

for variant in ['n', 's', 'm', 'l', 'x']:
    model = YOLO(f'yolo11{variant}.pt')
    model.export(format='coreml', imgsz=640)

    pt_file = Path(f'yolo11{variant}.pt')
    ml_dir = Path(f'yolo11{variant}.mlpackage')

    if pt_file.exists():
        shutil.move(str(pt_file), f'models/yolov11/yolo11{variant}.pt')
    if ml_dir.exists():
        shutil.move(str(ml_dir), f'models/yolov11/yolo11{variant}.mlpackage')
EOF
```

### 3. Prepare Test Video

Place a test video at `video/test_video.mp4`. You can use any video, or download a sample:

```bash
mkdir -p video
# Use your own video or download one
# The benchmark uses 640x640 input, so 720p+ video is recommended
```

### 4. Run Benchmarks

```bash
# ANE FP16 (baseline)
python3 run_benchmark.py

# Convert to INT8 and benchmark
python3 convert_to_int8.py
python3 run_benchmark_int8.py

# Convert to W8A8 and benchmark
python3 extract_calibration_frames.py  # Extract calibration data first
python3 convert_to_w8a8.py
python3 run_benchmark_w8a8.py

# GPU benchmarks
python3 run_benchmark_gpu.py      # FP32
python3 run_benchmark_gpu_fp16.py # FP16
```

## Test Configuration

Our benchmarks used the following configuration:

- **Platform**: macOS 26.1 (arm64)
- **Processor**: Apple M4
- **Input Size**: 640x640
- **Warmup Frames**: 10
- **Benchmark Frames**: 100
- **Test Video**: 15,477 frames @ 24 FPS

## Complete Results

### FPS by Backend and Precision

| Model | ANE FP16 | ANE INT8 | ANE W8A8 | GPU FP32 | GPU FP16 |
|-------|----------|----------|----------|----------|----------|
| **YOLOv8n** | 139.4 | 140.5 | 139.4 | 20.0 | 21.0 |
| **YOLOv8s** | 116.5 | 116.0 | 114.2 | 17.1 | 18.9 |
| **YOLOv8m** | 82.5 | 83.0 | 83.0 | 14.6 | 15.6 |
| **YOLOv8l** | 54.9 | 57.3 | 57.7 | 12.8 | 14.3 |
| **YOLOv8x** | 39.7 | 41.0 | 41.9 | 10.0 | 11.5 |
| **YOLOv11n** | 139.7 | 140.6 | 138.7 | 26.9 | 28.4 |
| **YOLOv11s** | 114.4 | 115.2 | 116.0 | 22.2 | 24.3 |
| **YOLOv11m** | 68.1 | 77.9 | 78.7 | 16.9 | 19.4 |
| **YOLOv11l** | 57.5 | 68.4 | 67.6 | 15.7 | 17.9 |
| **YOLOv11x** | 35.6 | 41.2 | 41.5 | 9.9 | 12.4 |

### Latency (ms) by Backend and Precision

| Model | ANE FP16 | ANE INT8 | ANE W8A8 | GPU FP32 | GPU FP16 |
|-------|----------|----------|----------|----------|----------|
| **YOLOv8n** | 7.2 | 7.1 | 7.2 | 50.1 | 47.6 |
| **YOLOv8s** | 8.6 | 8.6 | 8.8 | 58.7 | 52.9 |
| **YOLOv8m** | 12.1 | 12.0 | 12.0 | 68.5 | 64.2 |
| **YOLOv8l** | 18.2 | 17.4 | 17.3 | 78.0 | 69.9 |
| **YOLOv8x** | 25.2 | 24.4 | 23.9 | 99.6 | 87.2 |
| **YOLOv11n** | 7.2 | 7.1 | 7.2 | 37.2 | 35.2 |
| **YOLOv11s** | 8.7 | 8.7 | 8.6 | 45.0 | 41.1 |
| **YOLOv11m** | 14.7 | 12.8 | 12.7 | 59.1 | 51.7 |
| **YOLOv11l** | 17.4 | 14.6 | 14.8 | 63.6 | 55.9 |
| **YOLOv11x** | 28.1 | 24.3 | 24.1 | 100.7 | 81.0 |

## Quantization Details

### INT8 (Weight-Only) vs W8A8 (Full INT8)

| Quantization | Weights | Activations | Calibration Required | Best For |
|--------------|---------|-------------|---------------------|----------|
| **FP16** | 16-bit float | 16-bit float | No | Baseline, accuracy |
| **INT8** | 8-bit int | 16-bit float | No | All Apple Silicon |
| **W8A8** | 8-bit int | 8-bit int | Yes | M4/A17 Pro+ chips |

### Model Size Comparison

| Model | FP16 Size | INT8 Size | W8A8 Size | Compression |
|-------|-----------|-----------|-----------|-------------|
| YOLOv8n | 6.2 MB | 3.2 MB | 3.2 MB | ~1.95x |
| YOLOv8s | 21.4 MB | 10.8 MB | 10.8 MB | ~1.99x |
| YOLOv8m | 49.6 MB | 24.9 MB | 24.9 MB | ~1.99x |
| YOLOv8l | 83.5 MB | 41.9 MB | 41.9 MB | ~1.99x |
| YOLOv8x | 130.3 MB | 65.3 MB | 65.4 MB | ~2.00x |
| YOLOv11n | 5.2 MB | 2.7 MB | 2.7 MB | ~1.92x |
| YOLOv11s | 18.2 MB | 9.2 MB | 9.2 MB | ~1.98x |
| YOLOv11m | 38.6 MB | 19.4 MB | 19.4 MB | ~1.98x |
| YOLOv11l | 48.7 MB | 24.6 MB | 24.6 MB | ~1.98x |
| YOLOv11x | 108.9 MB | 54.7 MB | 54.8 MB | ~1.99x |

### Performance Improvement from Quantization

| Model | FP16 FPS | INT8 FPS | W8A8 FPS | INT8 Gain | W8A8 Gain |
|-------|----------|----------|----------|-----------|-----------|
| YOLOv8n | 139.4 | 140.5 | 139.4 | +0.8% | +0.0% |
| YOLOv8s | 116.5 | 116.0 | 114.2 | -0.4% | -2.0% |
| YOLOv8m | 82.5 | 83.0 | 83.0 | +0.6% | +0.6% |
| YOLOv8l | 54.9 | 57.3 | 57.7 | +4.4% | +5.1% |
| YOLOv8x | 39.7 | 41.0 | 41.9 | +3.3% | +5.5% |
| YOLOv11n | 139.7 | 140.6 | 138.7 | +0.6% | -0.7% |
| YOLOv11s | 114.4 | 115.2 | 116.0 | +0.7% | +1.4% |
| YOLOv11m | 68.1 | 77.9 | 78.7 | **+14.4%** | **+15.6%** |
| YOLOv11l | 57.5 | 68.4 | 67.6 | **+19.0%** | **+17.6%** |
| YOLOv11x | 35.6 | 41.2 | 41.5 | **+15.7%** | **+16.6%** |

## ANE vs GPU Analysis

### Why is ANE So Much Faster?

The Apple Neural Engine is purpose-built for neural network inference:

1. **Dedicated Hardware**: ANE has specialized matrix multiply units optimized for convolutions
2. **Memory Bandwidth**: ANE has dedicated high-bandwidth memory access
3. **Power Efficiency**: ANE achieves higher performance per watt
4. **Batch Optimization**: ANE is optimized for the batch size of 1 common in real-time inference

### Speedup: ANE FP16 vs GPU FP16

| Model | ANE FP16 | GPU FP16 | ANE Speedup |
|-------|----------|----------|-------------|
| YOLOv8n | 139.4 | 21.0 | **6.6x** |
| YOLOv8s | 116.5 | 18.9 | **6.2x** |
| YOLOv8m | 82.5 | 15.6 | **5.3x** |
| YOLOv8l | 54.9 | 14.3 | **3.8x** |
| YOLOv8x | 39.7 | 11.5 | **3.5x** |
| YOLOv11n | 139.7 | 28.4 | **4.9x** |
| YOLOv11s | 114.4 | 24.3 | **4.7x** |
| YOLOv11m | 68.1 | 19.4 | **3.5x** |
| YOLOv11l | 57.5 | 17.9 | **3.2x** |
| YOLOv11x | 35.6 | 12.4 | **2.9x** |

**Observations:**
- Smaller models see larger speedups (6-7x for nano vs 3x for extra-large)
- YOLOv11 has slightly lower ANE speedup due to better GPU optimization
- The speedup diminishes for larger models as memory bandwidth becomes limiting

## Key Insights

### ANE Performance Characteristics
- **Nano models achieve ~140 FPS** - suitable for 4K real-time processing
- **Consistent low latency** with P95 within ~10% of P50 (very stable)
- **INT8 benefits larger models more** - up to 19% speedup for YOLOv11l

### GPU (Metal/MPS) Performance Characteristics
- **3-7x slower than ANE** for single-image inference
- **Higher latency variance** - P95 can be 2x the P50
- **FP16 provides ~10-20% speedup** over FP32

### Quantization Insights
- **INT8 and W8A8 perform similarly** on M4 - both use per-channel quantization
- **Best results with YOLOv11 m/l/x models** - architecture benefits more from INT8
- **~2x model size reduction** with no significant accuracy loss
- **W8A8 may provide additional benefits** on future Apple Silicon with enhanced INT8 compute

## Benchmark Scripts Reference

| Script | Description | Backend | Precision |
|--------|-------------|---------|-----------|
| `run_benchmark.py` | CoreML ANE benchmark | ANE | FP16 |
| `run_benchmark_int8.py` | INT8 quantized models | ANE | INT8 |
| `run_benchmark_w8a8.py` | W8A8 quantized models | ANE | W8A8 |
| `run_benchmark_gpu.py` | PyTorch MPS benchmark | GPU | FP32 |
| `run_benchmark_gpu_fp16.py` | PyTorch MPS half precision | GPU | FP16 |
| `convert_to_int8.py` | Weight-only INT8 conversion | - | INT8 |
| `convert_to_w8a8.py` | Full INT8 conversion | - | W8A8 |
| `extract_calibration_frames.py` | Extract frames for calibration | - | - |

## Directory Structure

```
benchmarks/
├── models/                    # Model files (not tracked in git)
│   ├── yolov8/               # YOLOv8 variants
│   ├── yolov11/              # YOLOv11 variants
│   └── yolov12/              # YOLOv12 variants (when available)
├── video/                     # Test video (not tracked in git)
├── calibration_images/        # Extracted frames for INT8 calibration
├── results/                   # JSON benchmark results
├── run_benchmark.py           # ANE FP16 benchmark
├── run_benchmark_int8.py      # ANE INT8 benchmark
├── run_benchmark_w8a8.py      # ANE W8A8 benchmark
├── run_benchmark_gpu.py       # GPU FP32 benchmark
├── run_benchmark_gpu_fp16.py  # GPU FP16 benchmark
├── convert_to_int8.py         # INT8 conversion
├── convert_to_w8a8.py         # W8A8 conversion
├── extract_calibration_frames.py
└── README.md
```

## Recommendations

| Use Case | Recommended Model | Backend | Precision |
|----------|-------------------|---------|-----------|
| Real-time (>100 FPS) | YOLOv8n/YOLOv11n | ANE | INT8 or W8A8 |
| Balanced speed/accuracy | YOLOv8s/YOLOv11s | ANE | INT8 or W8A8 |
| High accuracy | YOLOv8m/YOLOv11m | ANE | INT8 or W8A8 |
| Maximum accuracy | YOLOv8l/YOLOv11l | ANE | FP16 |
| Memory constrained | Any model | ANE | INT8 or W8A8 |

## Notes

- YOLOv12 models were not available in the Ultralytics repository at the time of testing
- INT8 uses weight-only quantization (INT8 weights, FP16 activations)
- W8A8 uses full quantization (INT8 weights + INT8 activations) with per-channel granularity
- GPU benchmarks use PyTorch MPS backend - CoreML GPU may differ
- All benchmarks run with warmup to ensure stable measurements
- Calibration data is extracted from the test video (100 frames)
