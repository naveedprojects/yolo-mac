#ifndef YOLOV12_PYTHON_INTERPRETER_H
#define YOLOV12_PYTHON_INTERPRETER_H

#ifdef YOLOV12_ENABLE_PYTHON

#include "yolov12/converter.h"
#include <string>

namespace yolov12 {
namespace python_bridge {

/**
 * @brief Initialize Python interpreter
 * @return true if successful
 */
bool initialize_python();

/**
 * @brief Finalize Python interpreter
 */
void finalize_python();

/**
 * @brief Check if required Python packages are available
 * @return true if coremltools, torch, and ultralytics are available
 */
bool check_python_packages();

/**
 * @brief Run conversion script
 */
ConversionResult run_conversion_script(
    const std::string& input_path,
    const std::string& output_path,
    const ConversionConfig& config);

/**
 * @brief Run advanced quantization script
 */
ConversionResult run_advanced_quantization(
    const std::string& input_path,
    const std::string& output_path,
    const ConversionConfig& config);

} // namespace python_bridge
} // namespace yolov12

#endif // YOLOV12_ENABLE_PYTHON

#endif // YOLOV12_PYTHON_INTERPRETER_H
