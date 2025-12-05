#ifndef YOLOV12_H
#define YOLOV12_H

/**
 * @file yolov12.h
 * @brief Main header for YOLOv12-Mac framework
 *
 * YOLOv12-Mac is a C++ framework for running YOLO object detection
 * models (YOLOv8, YOLOv11, YOLOv12) on Apple Silicon Macs with
 * optimized performance using Apple Neural Engine (ANE), GPU (Metal), or CPU.
 *
 * @section supported_models Supported Models
 * - YOLOv12 (n, s, m, l, x variants)
 * - YOLOv11 (n, s, m, l, x variants)
 * - YOLOv8  (n, s, m, l, x variants)
 *
 * @section features Features
 * - Pure C++ API with Objective-C++ internals
 * - CoreML and ONNX Runtime backends
 * - Support for ANE, GPU, and CPU execution
 * - Model conversion with FP16, INT8, INT4, W8A8 quantization
 * - Optional Python backend for advanced quantization
 *
 * @section quick_start Quick Start
 * @code{.cpp}
 * #include <yolov12/yolov12.h>
 *
 * // Load model with ANE preference
 * auto detector = yolov12::Detector::create("yolov12n.mlpackage",
 *     yolov12::DetectorConfig::for_ane());
 *
 * // Detect objects
 * auto image = yolov12::Image::from_file("test.jpg");
 * auto detections = detector->detect(image);
 *
 * for (const auto& det : detections) {
 *     std::cout << det.class_name << ": " << det.confidence << std::endl;
 * }
 * @endcode
 *
 * @section conversion Model Conversion
 * @code{.cpp}
 * yolov12::ConversionConfig config;
 * config.quantization = yolov12::QuantizationType::INT8;
 * config.calibration_data_path = "./images/";
 *
 * auto result = yolov12::Converter::convert_onnx(
 *     "yolov12n.onnx", "yolov12n.mlpackage", config);
 * @endcode
 */

// Version information
#define YOLOV12_VERSION_MAJOR 1
#define YOLOV12_VERSION_MINOR 0
#define YOLOV12_VERSION_PATCH 0
#define YOLOV12_VERSION_STRING "1.0.0"

// Include all public headers
#include "yolov12/types.h"
#include "yolov12/compute_unit.h"
#include "yolov12/converter.h"
#include "yolov12/detector.h"

namespace yolov12 {

/**
 * @brief Library version string
 */
constexpr const char* VERSION = YOLOV12_VERSION_STRING;

/**
 * @brief Library version as integer (major * 10000 + minor * 100 + patch)
 */
constexpr int VERSION_INT = YOLOV12_VERSION_MAJOR * 10000 +
                            YOLOV12_VERSION_MINOR * 100 +
                            YOLOV12_VERSION_PATCH;

/**
 * @brief Check if Python backend is available at runtime
 *
 * Returns true if the library was built with Python support AND
 * Python interpreter and required packages are available.
 *
 * @return true if Python backend can be used
 */
bool is_python_available();

/**
 * @brief Get list of supported quantization types
 *
 * Returns quantization types supported by the current build.
 * Pure C++ builds support fewer types than Python-enabled builds.
 *
 * @return Vector of supported quantization types
 */
std::vector<QuantizationType> get_supported_quantizations();

/**
 * @brief Get build configuration info
 * @return String describing build options
 */
std::string get_build_info();

/**
 * @brief Initialize the library
 *
 * Optional - automatically called on first use.
 * Can be called explicitly to control initialization timing.
 */
void initialize();

/**
 * @brief Shutdown and cleanup
 *
 * Optional - automatically called on program exit.
 * Releases any global resources.
 */
void shutdown();

/**
 * @brief Check if running on Apple Silicon
 * @return true if running on M1/M2/M3/M4/M5 chip
 */
bool is_apple_silicon();

/**
 * @brief Get detected Apple Silicon generation
 * @return Chip generation (1 for M1, 2 for M2, etc.) or 0 if not Apple Silicon
 */
int get_apple_silicon_generation();

/**
 * @brief Check if current hardware supports W8A8 acceleration
 *
 * W8A8 (INT8 weights + activations) is only accelerated on
 * M4/A17 Pro and later chips.
 *
 * @return true if W8A8 provides performance benefit
 */
bool supports_w8a8_acceleration();

// ==================== Error Handling ====================

/**
 * @brief Exception class for YOLOv12 errors
 */
class Exception : public std::runtime_error {
public:
    enum class Code {
        UNKNOWN,
        MODEL_NOT_FOUND,
        MODEL_LOAD_FAILED,
        INVALID_IMAGE,
        INFERENCE_FAILED,
        CONVERSION_FAILED,
        PYTHON_ERROR,
        UNSUPPORTED_QUANTIZATION,
        CALIBRATION_FAILED
    };

    Exception(Code code, const std::string& message)
        : std::runtime_error(message), code_(code) {}

    Code code() const { return code_; }

private:
    Code code_;
};

// ==================== Utility Functions ====================

/**
 * @brief Draw detections on image (if OpenCV available)
 *
 * This is a utility function that requires OpenCV. If OpenCV is not
 * available, this function does nothing.
 *
 * @param image_path Input image path
 * @param output_path Output image path
 * @param detections Detection results to draw
 * @param draw_labels Whether to draw class names and confidence
 * @return true if successful
 */
bool draw_detections(
    const std::string& image_path,
    const std::string& output_path,
    const std::vector<Detection>& detections,
    bool draw_labels = true
);

/**
 * @brief Save detections to JSON file
 * @param output_path Path to output JSON file
 * @param detections Detection results
 * @param image_width Original image width
 * @param image_height Original image height
 * @return true if successful
 */
bool save_detections_json(
    const std::string& output_path,
    const std::vector<Detection>& detections,
    int image_width,
    int image_height
);

/**
 * @brief Format detections as string
 * @param detections Detection results
 * @return Formatted string
 */
std::string format_detections(const std::vector<Detection>& detections);

} // namespace yolov12

#endif // YOLOV12_H
