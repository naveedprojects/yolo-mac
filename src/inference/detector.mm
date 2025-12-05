#include "yolov12/detector.h"
#include <stdexcept>
#include <algorithm>

namespace yolov12 {

// Check file extension to determine model type
static bool ends_with(const std::string& str, const std::string& suffix) {
    if (suffix.size() > str.size()) {
        return false;
    }
    return std::equal(suffix.rbegin(), suffix.rend(), str.rbegin(),
                      [](char a, char b) {
                          return std::tolower(a) == std::tolower(b);
                      });
}

std::unique_ptr<Detector> Detector::create(
    const std::string& model_path,
    const DetectorConfig& config) {

    // Determine backend from file extension
    if (ends_with(model_path, ".mlpackage") ||
        ends_with(model_path, ".mlmodel") ||
        ends_with(model_path, ".mlmodelc")) {
        return create_coreml(model_path, config);
    }
#ifdef YOLOV12_ENABLE_ONNX
    else if (ends_with(model_path, ".onnx")) {
        return create_onnx(model_path, config);
    }
#endif
    else {
        throw std::runtime_error(
            "Unsupported model format. Supported formats: "
            ".mlpackage, .mlmodel, .mlmodelc"
#ifdef YOLOV12_ENABLE_ONNX
            ", .onnx"
#endif
        );
    }
}

#ifndef YOLOV12_ENABLE_ONNX
// Stub for ONNX when not enabled
std::unique_ptr<Detector> Detector::create_onnx(
    const std::string& model_path,
    const DetectorConfig& config) {
    throw std::runtime_error("ONNX Runtime support not enabled. "
                             "Rebuild with -DYOLOV12_ENABLE_ONNX=ON");
}
#endif

} // namespace yolov12
