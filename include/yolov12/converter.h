#ifndef YOLOV12_CONVERTER_H
#define YOLOV12_CONVERTER_H

#include <string>
#include <vector>
#include <optional>
#include "types.h"

namespace yolov12 {

/**
 * @brief Quantization type for model conversion
 */
enum class QuantizationType {
    NONE,           ///< FP32 - no quantization
    FP16,           ///< 16-bit float (recommended for ANE)
    INT8,           ///< 8-bit integer, linear quantization
    INT4,           ///< 4-bit integer (Python backend only)
    W8A8,           ///< INT8 weights + INT8 activations (Python backend, M4+ only)
    PALETTIZE_4,    ///< 4-bit palettization/clustering (Python backend only)
    PALETTIZE_8,    ///< 8-bit palettization/clustering (Python backend only)
    MIXED           ///< Mixed precision per-layer (Python backend only)
};

/**
 * @brief Input model format
 */
enum class ModelFormat {
    UNKNOWN,        ///< Unknown format
    PYTORCH,        ///< PyTorch .pt file (requires Python backend)
    ONNX,           ///< ONNX .onnx file (pure C++)
    COREML          ///< CoreML .mlpackage or .mlmodel (already converted)
};

/**
 * @brief Quantization granularity
 */
enum class QuantizationGranularity {
    PER_TENSOR,     ///< One scale per tensor (fastest, least accurate)
    PER_CHANNEL,    ///< One scale per channel (recommended for ANE)
    PER_BLOCK       ///< One scale per block (best for INT4, GPU)
};

/**
 * @brief Convert QuantizationType to string
 */
inline const char* to_string(QuantizationType quant) {
    switch (quant) {
        case QuantizationType::NONE:        return "FP32";
        case QuantizationType::FP16:        return "FP16";
        case QuantizationType::INT8:        return "INT8";
        case QuantizationType::INT4:        return "INT4";
        case QuantizationType::W8A8:        return "W8A8";
        case QuantizationType::PALETTIZE_4: return "PALETTIZE_4";
        case QuantizationType::PALETTIZE_8: return "PALETTIZE_8";
        case QuantizationType::MIXED:       return "MIXED";
        default:                            return "UNKNOWN";
    }
}

/**
 * @brief Parse string to QuantizationType
 */
inline QuantizationType parse_quantization_type(const std::string& str) {
    std::string upper = str;
    for (auto& c : upper) {
        c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    }

    if (upper == "NONE" || upper == "FP32") return QuantizationType::NONE;
    if (upper == "FP16" || upper == "HALF") return QuantizationType::FP16;
    if (upper == "INT8") return QuantizationType::INT8;
    if (upper == "INT4") return QuantizationType::INT4;
    if (upper == "W8A8") return QuantizationType::W8A8;
    if (upper == "PALETTIZE_4" || upper == "PAL4") return QuantizationType::PALETTIZE_4;
    if (upper == "PALETTIZE_8" || upper == "PAL8") return QuantizationType::PALETTIZE_8;
    if (upper == "MIXED") return QuantizationType::MIXED;

    return QuantizationType::NONE;
}

/**
 * @brief Configuration for model conversion
 */
struct ConversionConfig {
    // Quantization settings
    QuantizationType quantization = QuantizationType::FP16;
    QuantizationGranularity granularity = QuantizationGranularity::PER_CHANNEL;
    std::string mixed_precision_config;  ///< Path to YAML for MIXED type

    // Calibration settings (for INT8, W8A8)
    std::string calibration_data_path;   ///< Directory with calibration images
    int calibration_samples = 128;       ///< Number of samples to use

    // Model input size
    int input_width = 640;
    int input_height = 640;
    int input_channels = 3;

    // Post-processing options
    bool include_nms = false;            ///< Include NMS in model
    float nms_iou_threshold = 0.45f;
    float nms_confidence_threshold = 0.25f;
    int nms_max_detections = 100;

    // Target deployment
    int minimum_ios_version = 15;        ///< iOS 17 required for W8A8

    // Output options
    bool overwrite_existing = false;
    bool verbose = false;

    /**
     * @brief Check if this config requires Python backend
     */
    bool requires_python() const {
        return quantization == QuantizationType::INT4 ||
               quantization == QuantizationType::W8A8 ||
               quantization == QuantizationType::PALETTIZE_4 ||
               quantization == QuantizationType::PALETTIZE_8 ||
               quantization == QuantizationType::MIXED;
    }
};

/**
 * @brief Result of model conversion
 */
struct ConversionResult {
    bool success = false;
    std::string output_path;
    std::string error_message;
    ModelInfo model_info;

    // Conversion statistics
    double original_size_mb = 0.0;
    double converted_size_mb = 0.0;
    double compression_ratio = 1.0;
    double conversion_time_seconds = 0.0;

    // Quantization details
    std::string quantization_applied;
    int num_quantized_layers = 0;
    int num_total_layers = 0;
};

/**
 * @brief Model converter class
 *
 * Provides static methods for converting YOLO models to optimized formats
 * for Apple Silicon. Supports both pure C++ (ONNX) and Python-enhanced
 * (PyTorch) conversion paths.
 */
class Converter {
public:
    /**
     * @brief Detect model format from file extension
     * @param path Path to model file
     * @return Detected ModelFormat
     */
    static ModelFormat detect_format(const std::string& path);

    /**
     * @brief Check if Python backend is available
     * @return true if Python and required packages are available
     */
    static bool is_python_available();

    /**
     * @brief Check if a quantization type requires Python backend
     * @param quant Quantization type
     * @return true if Python is required
     */
    static bool requires_python(QuantizationType quant);

    /**
     * @brief Get supported quantization types for current build
     * @return Vector of supported quantization types
     */
    static std::vector<QuantizationType> get_supported_quantizations();

    /**
     * @brief Convert model (auto-detects format and dispatches to appropriate backend)
     * @param input_path Path to input model (.pt, .onnx)
     * @param output_path Path for output model (.mlpackage)
     * @param config Conversion configuration
     * @return Conversion result
     */
    static ConversionResult convert(
        const std::string& input_path,
        const std::string& output_path,
        const ConversionConfig& config = {}
    );

    /**
     * @brief Convert ONNX model (pure C++)
     *
     * Supports FP16, INT8 quantization without Python.
     *
     * @param onnx_path Path to ONNX model
     * @param output_path Path for output model
     * @param config Conversion configuration
     * @return Conversion result
     */
    static ConversionResult convert_onnx(
        const std::string& onnx_path,
        const std::string& output_path,
        const ConversionConfig& config = {}
    );

    /**
     * @brief Convert PyTorch model (requires Python backend)
     *
     * Supports all quantization types including W8A8, INT4, mixed precision.
     *
     * @param pt_path Path to PyTorch model (.pt)
     * @param output_path Path for output model
     * @param config Conversion configuration
     * @return Conversion result
     */
    static ConversionResult convert_pytorch(
        const std::string& pt_path,
        const std::string& output_path,
        const ConversionConfig& config = {}
    );

    /**
     * @brief Quantize an existing CoreML model
     * @param input_path Path to CoreML model
     * @param output_path Path for quantized model
     * @param config Quantization configuration
     * @return Conversion result
     */
    static ConversionResult quantize(
        const std::string& input_path,
        const std::string& output_path,
        const ConversionConfig& config
    );

    /**
     * @brief Validate a converted model
     * @param model_path Path to model
     * @return Validation result with model info
     */
    static ConversionResult validate(const std::string& model_path);
};

} // namespace yolov12

#endif // YOLOV12_CONVERTER_H
