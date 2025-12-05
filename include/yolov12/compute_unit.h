#ifndef YOLOV12_COMPUTE_UNIT_H
#define YOLOV12_COMPUTE_UNIT_H

#include <string>
#include <stdexcept>

namespace yolov12 {

/**
 * @brief Compute unit selection for model inference
 *
 * These map to Apple's MLComputeUnits values for CoreML,
 * and equivalent settings for ONNX Runtime's CoreML Execution Provider.
 */
enum class ComputeUnit {
    CPU_ONLY,       ///< Force CPU only (BNNS on Apple Silicon)
    GPU_ONLY,       ///< Prefer GPU (Metal)
    ANE_ONLY,       ///< Prefer Apple Neural Engine (best for M-series)
    CPU_AND_GPU,    ///< Allow CPU and GPU
    CPU_AND_ANE,    ///< Allow CPU and ANE
    ALL             ///< Let system decide (recommended)
};

/**
 * @brief Convert ComputeUnit to human-readable string
 * @param unit The compute unit enum value
 * @return String representation
 */
inline const char* to_string(ComputeUnit unit) {
    switch (unit) {
        case ComputeUnit::CPU_ONLY:     return "CPU_ONLY";
        case ComputeUnit::GPU_ONLY:     return "GPU_ONLY";
        case ComputeUnit::ANE_ONLY:     return "ANE_ONLY";
        case ComputeUnit::CPU_AND_GPU:  return "CPU_AND_GPU";
        case ComputeUnit::CPU_AND_ANE:  return "CPU_AND_ANE";
        case ComputeUnit::ALL:          return "ALL";
        default:                        return "UNKNOWN";
    }
}

/**
 * @brief Parse string to ComputeUnit
 * @param str String representation (case-insensitive)
 * @return ComputeUnit enum value
 * @throws std::invalid_argument if string is not recognized
 */
inline ComputeUnit parse_compute_unit(const std::string& str) {
    // Convert to uppercase for comparison
    std::string upper = str;
    for (auto& c : upper) {
        c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    }

    if (upper == "CPU_ONLY" || upper == "CPU") {
        return ComputeUnit::CPU_ONLY;
    } else if (upper == "GPU_ONLY" || upper == "GPU") {
        return ComputeUnit::GPU_ONLY;
    } else if (upper == "ANE_ONLY" || upper == "ANE" || upper == "NEURAL_ENGINE") {
        return ComputeUnit::ANE_ONLY;
    } else if (upper == "CPU_AND_GPU") {
        return ComputeUnit::CPU_AND_GPU;
    } else if (upper == "CPU_AND_ANE") {
        return ComputeUnit::CPU_AND_ANE;
    } else if (upper == "ALL" || upper == "AUTO") {
        return ComputeUnit::ALL;
    } else {
        throw std::invalid_argument("Unknown compute unit: " + str);
    }
}

/**
 * @brief Get description of compute unit capabilities
 * @param unit The compute unit
 * @return Description string
 */
inline const char* get_compute_unit_description(ComputeUnit unit) {
    switch (unit) {
        case ComputeUnit::CPU_ONLY:
            return "CPU only - uses BNNS (Basic Neural Network Subroutines)";
        case ComputeUnit::GPU_ONLY:
            return "GPU only - uses Metal Performance Shaders";
        case ComputeUnit::ANE_ONLY:
            return "Apple Neural Engine - best for M-series chips, requires FP16 model";
        case ComputeUnit::CPU_AND_GPU:
            return "CPU and GPU - hybrid execution";
        case ComputeUnit::CPU_AND_ANE:
            return "CPU and ANE - hybrid with Neural Engine priority";
        case ComputeUnit::ALL:
            return "All compute units - system chooses optimal (recommended)";
        default:
            return "Unknown compute unit";
    }
}

/**
 * @brief Check if compute unit supports INT8 acceleration
 * @param unit The compute unit
 * @param chip_generation M-series chip generation (1, 2, 3, 4, 5)
 * @return true if INT8 compute is accelerated
 */
inline bool supports_int8_acceleration(ComputeUnit unit, int chip_generation) {
    // Only M4 (generation 4) and later support native INT8 on ANE
    if (chip_generation >= 4) {
        return (unit == ComputeUnit::ANE_ONLY ||
                unit == ComputeUnit::CPU_AND_ANE ||
                unit == ComputeUnit::ALL);
    }
    // GPU supports INT8 on all generations (via Metal)
    return (unit == ComputeUnit::GPU_ONLY || unit == ComputeUnit::CPU_AND_GPU);
}

} // namespace yolov12

#endif // YOLOV12_COMPUTE_UNIT_H
