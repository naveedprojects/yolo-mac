# YOLOv12-Mac Framework - Execution Plan

A C++ framework for converting and running YOLOv12 models on macOS with Apple Silicon optimization (ANE/GPU/CPU).

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     YOLOv12-Mac Framework                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ PATH 1: Pure C++ (ONNX-based)                               ││
│  │ ─────────────────────────────────────────                   ││
│  │ • User provides .onnx file                                  ││
│  │ • ONNX Runtime + CoreML Execution Provider                  ││
│  │ • INT8/FP16 via ONNX quantization                          ││
│  │ • No Python dependency                                      ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                  │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ PATH 2: C++ with Python Backend (Full Features)             ││
│  │ ─────────────────────────────────────────────               ││
│  │ • User provides .pt file                                    ││
│  │ • Full coremltools quantization (INT4, W8A8, mixed)        ││
│  │ • Embedded Python or subprocess                             ││
│  │ • Maximum optimization control                              ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                  │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ INFERENCE: Pure C++ / Obj-C++ (Both paths converge here)    ││
│  │ ───────────────────────────────────────────────────────     ││
│  │ • Core ML / ONNX Runtime inference                          ││
│  │ • ANE / GPU / CPU selection                                 ││
│  │ • No Python at inference time                               ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

---

## Quantization Support Matrix

| Method | Bits | Pure C++ | Python Backend | ANE Support |
|--------|------|----------|----------------|-------------|
| FP32 | 32 | ✅ | ✅ | ⚠️ Converted to FP16 |
| FP16 | 16 | ✅ | ✅ | ✅ All chips |
| Linear INT8 | 8 | ✅ | ✅ | ✅ All chips (size), M4+ (speed) |
| W8A8 | 8+8 | ❌ | ✅ | ✅ M4/A17 Pro only |
| Linear INT4 | 4 | ❌ | ✅ | ⚠️ GPU preferred |
| Palettization | 1-8 | ❌ | ✅ | ✅ All chips |
| Mixed Precision | varies | ❌ | ✅ | ✅ Depends on config |

### Hardware Compatibility

| Chip | ANE TOPS | INT8 Compute | Best Quantization |
|------|----------|--------------|-------------------|
| M1 | ~11 | ❌ Size only | FP16 |
| M2 | ~16 | ❌ Size only | FP16 |
| M3 | 18 | ❌ Size only | FP16 |
| M4 | 38 | ✅ Native | W8A8 |
| M5 | 4x M4 | ✅ Native | W8A8 |

---

## Project Structure

```
yolov12-mac/
├── CMakeLists.txt
├── PLAN.md
├── README.md
├── include/
│   └── yolov12/
│       ├── yolov12.h            # Main header (includes all)
│       ├── detector.h           # Inference API
│       ├── converter.h          # Conversion API
│       ├── types.h              # Detection, Image, Config types
│       └── compute_unit.h       # ComputeUnit enum
├── src/
│   ├── inference/               # Pure C++ / Obj-C++
│   │   ├── detector.mm          # Main detector implementation
│   │   ├── detector_coreml.mm   # CoreML backend
│   │   ├── detector_onnx.cpp    # ONNX Runtime backend
│   │   ├── model_loader.mm      # CoreML model loading
│   │   ├── preprocessor.mm      # Image preprocessing
│   │   └── postprocessor.cpp    # NMS, box decoding
│   ├── conversion/
│   │   ├── converter.cpp        # Conversion dispatcher
│   │   ├── onnx_converter.cpp   # Pure C++ (ONNX Runtime)
│   │   ├── pytorch_converter.cpp # C++ with embedded Python
│   │   └── quantization/
│   │       ├── quantizer.cpp        # Quantization dispatcher
│   │       ├── onnx_quantizer.cpp   # Pure C++ ONNX quantization
│   │       └── coreml_quantizer.cpp # Python bridge for coremltools
│   └── python_bridge/           # Optional Python integration
│       ├── python_interpreter.cpp
│       ├── python_interpreter.h
│       └── scripts/
│           ├── convert_model.py
│           ├── quantize_model.py
│           └── calibrate.py
├── third_party/
│   └── CMakeLists.txt           # Third-party dependencies
├── tools/
│   ├── convert_cli.cpp          # Command-line converter
│   └── benchmark_cli.cpp        # Benchmarking tool
├── examples/
│   ├── basic_detection.cpp      # Simple usage example
│   ├── video_detection.cpp      # Video processing example
│   ├── batch_detection.cpp      # Batch processing example
│   └── custom_quantization.cpp  # Advanced quantization example
├── python/                      # Standalone Python tools
│   ├── setup.py
│   └── yolov12_converter/
│       ├── __init__.py
│       ├── convert.py
│       ├── quantize.py
│       └── calibrate.py
└── tests/
    ├── test_detector.cpp
    ├── test_converter.cpp
    ├── test_quantization.cpp
    └── test_preprocessing.cpp
```

---

## C++ API Design

### Core Types (`types.h`)

```cpp
namespace yolov12 {

// Bounding box detection result
struct Detection {
    int class_id;
    float confidence;
    float x;      // Normalized [0,1]
    float y;      // Normalized [0,1]
    float width;  // Normalized [0,1]
    float height; // Normalized [0,1]
    std::string class_name;
};

// Input image wrapper
struct Image {
    const uint8_t* data;
    int width;
    int height;
    int channels;  // 3 for RGB, 4 for RGBA

    // Factory methods
    static Image from_file(const std::string& path);
    static Image from_buffer(const uint8_t* data, int w, int h, int c);
};

// Model metadata
struct ModelInfo {
    std::string name;
    int input_width;
    int input_height;
    int num_classes;
    std::vector<std::string> class_names;
    std::string quantization_type;
    std::string compute_unit;
};

} // namespace yolov12
```

### Compute Unit (`compute_unit.h`)

```cpp
namespace yolov12 {

enum class ComputeUnit {
    CPU_ONLY,       // Force CPU (BNNS)
    GPU_ONLY,       // Force GPU (Metal)
    ANE_ONLY,       // Force Apple Neural Engine
    CPU_AND_GPU,    // CPU + GPU hybrid
    CPU_AND_ANE,    // CPU + ANE hybrid
    ALL             // Let Core ML decide (recommended)
};

// Convert to string for logging
const char* to_string(ComputeUnit unit);

// Parse from string (for CLI)
ComputeUnit parse_compute_unit(const std::string& str);

} // namespace yolov12
```

### Conversion API (`converter.h`)

```cpp
namespace yolov12 {

enum class QuantizationType {
    NONE,           // FP32
    FP16,           // 16-bit float
    INT8,           // 8-bit integer (linear)
    INT4,           // 4-bit integer (Python backend only)
    W8A8,           // INT8 weights + activations (Python backend only)
    PALETTIZE_4,    // 4-bit palettization (Python backend only)
    PALETTIZE_8,    // 8-bit palettization (Python backend only)
    MIXED           // Mixed precision (Python backend only)
};

enum class ModelFormat {
    PYTORCH,        // .pt file (requires Python)
    ONNX,           // .onnx file (pure C++)
    COREML          // .mlpackage (already converted)
};

struct ConversionConfig {
    // Quantization settings
    QuantizationType quantization = QuantizationType::FP16;
    std::string mixed_precision_config;  // Path to YAML for MIXED type

    // Calibration (for INT8, W8A8)
    std::string calibration_data_path;
    int calibration_samples = 128;

    // Model input size
    int input_width = 640;
    int input_height = 640;

    // Post-processing
    bool include_nms = false;
    float nms_iou_threshold = 0.45f;
    float nms_confidence_threshold = 0.25f;

    // Target deployment
    int minimum_ios_version = 15;  // iOS 17 required for W8A8

    // Output
    bool overwrite_existing = false;
};

struct ConversionResult {
    bool success;
    std::string output_path;
    std::string error_message;
    ModelInfo model_info;

    // Conversion statistics
    double original_size_mb;
    double converted_size_mb;
    double compression_ratio;
};

class Converter {
public:
    // Detect input format automatically
    static ModelFormat detect_format(const std::string& path);

    // Check if quantization type requires Python backend
    static bool requires_python(QuantizationType quant);

    // Convert model (dispatches to appropriate backend)
    static ConversionResult convert(
        const std::string& input_path,
        const std::string& output_path,
        const ConversionConfig& config = {}
    );

    // Pure C++ conversion (ONNX only)
    static ConversionResult convert_onnx(
        const std::string& onnx_path,
        const std::string& output_path,
        const ConversionConfig& config = {}
    );

    // Python-enhanced conversion (full features)
    static ConversionResult convert_pytorch(
        const std::string& pt_path,
        const std::string& output_path,
        const ConversionConfig& config = {}
    );
};

} // namespace yolov12
```

### Inference API (`detector.h`)

```cpp
namespace yolov12 {

struct DetectorConfig {
    ComputeUnit compute_unit = ComputeUnit::ALL;
    float confidence_threshold = 0.25f;
    float iou_threshold = 0.45f;
    int max_detections = 100;
    bool use_async = false;  // Async inference
};

class Detector {
public:
    virtual ~Detector() = default;

    // Factory method - auto-detects backend from file extension
    static std::unique_ptr<Detector> create(
        const std::string& model_path,
        const DetectorConfig& config = {}
    );

    // Single image detection
    virtual std::vector<Detection> detect(const Image& image) = 0;

    // Batch detection (more efficient for multiple images)
    virtual std::vector<std::vector<Detection>> detect_batch(
        const std::vector<Image>& images
    ) = 0;

    // Async detection with callback
    virtual void detect_async(
        const Image& image,
        std::function<void(std::vector<Detection>)> callback
    ) = 0;

    // Model information
    virtual ModelInfo get_model_info() const = 0;

    // Get active compute unit (may differ from requested)
    virtual ComputeUnit get_active_compute_unit() const = 0;

    // Warm up model (first inference is slower)
    virtual void warmup(int iterations = 1) = 0;
};

} // namespace yolov12
```

### Main Header (`yolov12.h`)

```cpp
#ifndef YOLOV12_H
#define YOLOV12_H

#include "yolov12/types.h"
#include "yolov12/compute_unit.h"
#include "yolov12/converter.h"
#include "yolov12/detector.h"

namespace yolov12 {

// Library version
constexpr const char* VERSION = "1.0.0";

// Check if Python backend is available
bool is_python_available();

// Get supported quantization types for current build
std::vector<QuantizationType> get_supported_quantizations();

// Initialize library (optional, auto-called on first use)
void initialize();

// Cleanup (optional, auto-called on exit)
void shutdown();

} // namespace yolov12

#endif // YOLOV12_H
```

---

## Execution Phases

### Phase 1: Project Setup & Core Types
- [x] Create folder structure
- [ ] Create CMakeLists.txt with build options
- [ ] Implement core types (types.h, compute_unit.h)
- [ ] Set up third-party dependencies (ONNX Runtime)

### Phase 2: Inference Engine (Pure C++)
- [ ] Implement CoreML model loader (Obj-C++)
- [ ] Implement ONNX Runtime backend
- [ ] Implement image preprocessing
- [ ] Implement post-processing (NMS, box decoding)
- [ ] Create Detector factory and interface

### Phase 3: Pure C++ Conversion (ONNX Path)
- [ ] Implement ONNX to CoreML conversion using ONNX Runtime
- [ ] Implement ONNX-based INT8 quantization
- [ ] Implement ONNX-based FP16 quantization
- [ ] Create conversion CLI tool

### Phase 4: Python Backend Integration
- [ ] Implement Python interpreter embedding
- [ ] Create Python conversion scripts (coremltools)
- [ ] Implement W8A8 quantization
- [ ] Implement INT4 quantization
- [ ] Implement mixed precision support
- [ ] Implement calibration data handling

### Phase 5: Tools & Examples
- [ ] Create command-line converter tool
- [ ] Create benchmarking tool
- [ ] Write basic detection example
- [ ] Write video processing example
- [ ] Write batch processing example

### Phase 6: Testing & Documentation
- [ ] Unit tests for all components
- [ ] Integration tests
- [ ] Performance benchmarks
- [ ] API documentation
- [ ] Usage guide

---

## Build Configuration

### CMake Options

```cmake
# Build options
option(YOLOV12_BUILD_SHARED "Build shared library" ON)
option(YOLOV12_BUILD_STATIC "Build static library" OFF)
option(YOLOV12_BUILD_TOOLS "Build CLI tools" ON)
option(YOLOV12_BUILD_EXAMPLES "Build examples" ON)
option(YOLOV12_BUILD_TESTS "Build tests" ON)

# Backend options
option(YOLOV12_ENABLE_COREML "Enable CoreML backend" ON)
option(YOLOV12_ENABLE_ONNX "Enable ONNX Runtime backend" ON)
option(YOLOV12_ENABLE_PYTHON "Enable Python backend for advanced quantization" ON)

# Pure C++ mode (disables Python entirely)
option(YOLOV12_PURE_CPP "Build pure C++ version only (no Python)" OFF)
```

### Build Commands

```bash
# Full build (with Python support)
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(sysctl -n hw.ncpu)

# Pure C++ build (no Python dependency)
cmake .. -DCMAKE_BUILD_TYPE=Release -DYOLOV12_PURE_CPP=ON
make -j$(sysctl -n hw.ncpu)
```

---

## Dependencies

| Dependency | Version | Purpose | Required |
|------------|---------|---------|----------|
| CMake | 3.20+ | Build system | Yes |
| Xcode | 15+ | Compiler + Frameworks | Yes |
| ONNX Runtime | 1.16+ | ONNX inference + CoreML EP | Yes |
| Python | 3.9+ | Advanced quantization | Optional |
| coremltools | 8.0+ | CoreML conversion | Optional |
| ultralytics | 8.0+ | YOLO model handling | Optional |
| pybind11 | 2.11+ | Python embedding | Optional |
| GoogleTest | 1.14+ | Testing | Optional |

---

## Usage Examples

### Basic Detection (C++)

```cpp
#include <yolov12/yolov12.h>
#include <iostream>

int main() {
    // Load model with ANE preference
    yolov12::DetectorConfig config;
    config.compute_unit = yolov12::ComputeUnit::ANE_ONLY;
    config.confidence_threshold = 0.5f;

    auto detector = yolov12::Detector::create("yolov12n.mlpackage", config);

    // Load and detect
    auto image = yolov12::Image::from_file("test.jpg");
    auto detections = detector->detect(image);

    // Print results
    for (const auto& det : detections) {
        std::cout << det.class_name << ": " << det.confidence
                  << " at (" << det.x << ", " << det.y << ")\n";
    }

    return 0;
}
```

### Model Conversion (C++)

```cpp
#include <yolov12/yolov12.h>
#include <iostream>

int main() {
    yolov12::ConversionConfig config;
    config.quantization = yolov12::QuantizationType::INT8;
    config.calibration_data_path = "./calibration_images/";
    config.calibration_samples = 128;
    config.input_width = 640;
    config.input_height = 640;

    // Convert ONNX model (pure C++)
    auto result = yolov12::Converter::convert_onnx(
        "yolov12n.onnx",
        "yolov12n_int8.mlpackage",
        config
    );

    if (result.success) {
        std::cout << "Conversion successful!\n";
        std::cout << "Size: " << result.original_size_mb << " MB -> "
                  << result.converted_size_mb << " MB\n";
        std::cout << "Compression: " << result.compression_ratio << "x\n";
    } else {
        std::cerr << "Error: " << result.error_message << "\n";
    }

    return 0;
}
```

### CLI Usage

```bash
# Convert with FP16 (pure C++)
./yolov12-convert --input model.onnx --output model.mlpackage --quantize fp16

# Convert with INT8 + calibration (pure C++)
./yolov12-convert --input model.onnx --output model.mlpackage \
    --quantize int8 --calibration-data ./images/ --samples 128

# Convert PyTorch with W8A8 (requires Python)
./yolov12-convert --input model.pt --output model.mlpackage \
    --quantize w8a8 --calibration-data ./images/ --target-ios 17

# Benchmark model
./yolov12-benchmark --model model.mlpackage --compute-unit ane --iterations 100
```

---

## Notes

### Known Limitations

1. **No direct C++ API for Core ML conversion** - Apple only provides Python tools
2. **W8A8 requires iOS 17+ / macOS 14+** - Won't work on older devices
3. **ANE programming is indirect** - Must go through Core ML, no direct API
4. **ONNX → CoreML may lose some optimizations** - Direct PyTorch → CoreML is better

### Workarounds

1. **Pure C++ users**: Export to ONNX first, then use our ONNX converter
2. **Maximum optimization**: Use Python backend for coremltools access
3. **Fallback handling**: Library auto-falls back GPU→CPU if ANE unavailable

### Performance Tips

1. Use FP16 for best ANE compatibility across all M-series chips
2. Use W8A8 on M4/A17 Pro for maximum performance
3. Warm up model before benchmarking (first inference is slow)
4. Use batch processing for offline workloads
5. Profile with Xcode Instruments to verify ANE utilization
