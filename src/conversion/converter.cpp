#include "yolov12/converter.h"
#include <fstream>
#include <filesystem>
#include <chrono>
#include <algorithm>

namespace fs = std::filesystem;

namespace yolov12 {

ModelFormat Converter::detect_format(const std::string& path) {
    std::string lower_path = path;
    std::transform(lower_path.begin(), lower_path.end(), lower_path.begin(), ::tolower);

    if (lower_path.find(".pt") != std::string::npos ||
        lower_path.find(".pth") != std::string::npos) {
        return ModelFormat::PYTORCH;
    } else if (lower_path.find(".onnx") != std::string::npos) {
        return ModelFormat::ONNX;
    } else if (lower_path.find(".mlpackage") != std::string::npos ||
               lower_path.find(".mlmodel") != std::string::npos) {
        return ModelFormat::COREML;
    }

    return ModelFormat::UNKNOWN;
}

bool Converter::is_python_available() {
#ifdef YOLOV12_ENABLE_PYTHON
    // Check if Python interpreter and required packages are available
    // This would actually try to initialize Python and import coremltools
    // For now, return compile-time flag
    return true;
#else
    return false;
#endif
}

bool Converter::requires_python(QuantizationType quant) {
    switch (quant) {
        case QuantizationType::INT4:
        case QuantizationType::W8A8:
        case QuantizationType::PALETTIZE_4:
        case QuantizationType::PALETTIZE_8:
        case QuantizationType::MIXED:
            return true;
        default:
            return false;
    }
}

std::vector<QuantizationType> Converter::get_supported_quantizations() {
    std::vector<QuantizationType> supported = {
        QuantizationType::NONE,
        QuantizationType::FP16,
        QuantizationType::INT8
    };

#ifdef YOLOV12_ENABLE_PYTHON
    supported.push_back(QuantizationType::INT4);
    supported.push_back(QuantizationType::W8A8);
    supported.push_back(QuantizationType::PALETTIZE_4);
    supported.push_back(QuantizationType::PALETTIZE_8);
    supported.push_back(QuantizationType::MIXED);
#endif

    return supported;
}

ConversionResult Converter::convert(
    const std::string& input_path,
    const std::string& output_path,
    const ConversionConfig& config) {

    ConversionResult result;

    // Check input file exists
    if (!fs::exists(input_path)) {
        result.success = false;
        result.error_message = "Input file not found: " + input_path;
        return result;
    }

    // Check if output exists and overwrite not set
    if (fs::exists(output_path) && !config.overwrite_existing) {
        result.success = false;
        result.error_message = "Output file already exists: " + output_path +
                               " (use overwrite_existing=true to replace)";
        return result;
    }

    // Check if quantization requires Python
    if (config.requires_python() && !is_python_available()) {
        result.success = false;
        result.error_message = "Quantization type " +
                               std::string(to_string(config.quantization)) +
                               " requires Python backend, which is not available";
        return result;
    }

    // Detect format and dispatch
    ModelFormat format = detect_format(input_path);

    auto start = std::chrono::high_resolution_clock::now();

    switch (format) {
        case ModelFormat::ONNX:
            result = convert_onnx(input_path, output_path, config);
            break;

        case ModelFormat::PYTORCH:
            result = convert_pytorch(input_path, output_path, config);
            break;

        case ModelFormat::COREML:
            // Already in CoreML format, just quantize if needed
            if (config.quantization != QuantizationType::NONE) {
                result = quantize(input_path, output_path, config);
            } else {
                // Just copy
                try {
                    fs::copy(input_path, output_path,
                             fs::copy_options::overwrite_existing);
                    result.success = true;
                    result.output_path = output_path;
                } catch (const std::exception& e) {
                    result.success = false;
                    result.error_message = e.what();
                }
            }
            break;

        default:
            result.success = false;
            result.error_message = "Unknown model format: " + input_path;
            break;
    }

    auto end = std::chrono::high_resolution_clock::now();
    result.conversion_time_seconds = std::chrono::duration<double>(end - start).count();

    // Calculate sizes
    if (result.success && fs::exists(input_path) && fs::exists(output_path)) {
        result.original_size_mb = static_cast<double>(fs::file_size(input_path)) / (1024 * 1024);

        // For directories (.mlpackage), calculate total size
        if (fs::is_directory(output_path)) {
            size_t total_size = 0;
            for (const auto& entry : fs::recursive_directory_iterator(output_path)) {
                if (fs::is_regular_file(entry)) {
                    total_size += fs::file_size(entry);
                }
            }
            result.converted_size_mb = static_cast<double>(total_size) / (1024 * 1024);
        } else {
            result.converted_size_mb = static_cast<double>(fs::file_size(output_path)) / (1024 * 1024);
        }

        if (result.converted_size_mb > 0) {
            result.compression_ratio = result.original_size_mb / result.converted_size_mb;
        }
    }

    return result;
}

#ifndef YOLOV12_ENABLE_ONNX
ConversionResult Converter::convert_onnx(
    const std::string& onnx_path,
    const std::string& output_path,
    const ConversionConfig& config) {

    ConversionResult result;
    result.success = false;
    result.error_message = "ONNX conversion not available. "
                           "Rebuild with -DYOLOV12_ENABLE_ONNX=ON";
    return result;
}
#endif

#ifndef YOLOV12_ENABLE_PYTHON
ConversionResult Converter::convert_pytorch(
    const std::string& pt_path,
    const std::string& output_path,
    const ConversionConfig& config) {

    ConversionResult result;
    result.success = false;
    result.error_message = "PyTorch conversion requires Python backend. "
                           "Rebuild with -DYOLOV12_ENABLE_PYTHON=ON";
    return result;
}

ConversionResult Converter::quantize(
    const std::string& input_path,
    const std::string& output_path,
    const ConversionConfig& config) {

    ConversionResult result;

    // Only FP16 quantization is available without Python
    if (config.quantization != QuantizationType::FP16) {
        result.success = false;
        result.error_message = "Only FP16 quantization available without Python backend. "
                               "For " + std::string(to_string(config.quantization)) +
                               ", rebuild with -DYOLOV12_ENABLE_PYTHON=ON";
        return result;
    }

    // FP16 can be done during ONNX conversion
    result.success = false;
    result.error_message = "Post-conversion quantization requires Python backend";
    return result;
}
#endif

ConversionResult Converter::validate(const std::string& model_path) {
    ConversionResult result;

    if (!fs::exists(model_path)) {
        result.success = false;
        result.error_message = "Model file not found: " + model_path;
        return result;
    }

    ModelFormat format = detect_format(model_path);

    if (format == ModelFormat::UNKNOWN) {
        result.success = false;
        result.error_message = "Unknown model format";
        return result;
    }

    // Basic validation passed
    result.success = true;
    result.output_path = model_path;

    // Get file size
    if (fs::is_directory(model_path)) {
        size_t total_size = 0;
        for (const auto& entry : fs::recursive_directory_iterator(model_path)) {
            if (fs::is_regular_file(entry)) {
                total_size += fs::file_size(entry);
            }
        }
        result.converted_size_mb = static_cast<double>(total_size) / (1024 * 1024);
    } else {
        result.converted_size_mb = static_cast<double>(fs::file_size(model_path)) / (1024 * 1024);
    }

    return result;
}

} // namespace yolov12
