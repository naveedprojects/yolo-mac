#ifdef YOLOV12_ENABLE_PYTHON

#include "yolov12/converter.h"
#include <Python.h>
#include <string>
#include <stdexcept>
#include <mutex>
#include <fstream>
#include <sstream>

namespace yolov12 {
namespace python_bridge {

static std::mutex python_mutex;
static bool python_initialized = false;

/**
 * @brief Initialize Python interpreter
 */
bool initialize_python() {
    std::lock_guard<std::mutex> lock(python_mutex);

    if (python_initialized) {
        return true;
    }

    if (!Py_IsInitialized()) {
        Py_Initialize();

        if (!Py_IsInitialized()) {
            return false;
        }
    }

    python_initialized = true;
    return true;
}

/**
 * @brief Finalize Python interpreter
 */
void finalize_python() {
    std::lock_guard<std::mutex> lock(python_mutex);

    if (python_initialized && Py_IsInitialized()) {
        Py_Finalize();
        python_initialized = false;
    }
}

/**
 * @brief Check if required Python packages are available
 */
bool check_python_packages() {
    if (!initialize_python()) {
        return false;
    }

    std::lock_guard<std::mutex> lock(python_mutex);

    // Try to import required packages
    PyObject* coremltools = PyImport_ImportModule("coremltools");
    if (!coremltools) {
        PyErr_Clear();
        return false;
    }
    Py_DECREF(coremltools);

    PyObject* torch = PyImport_ImportModule("torch");
    if (!torch) {
        PyErr_Clear();
        return false;
    }
    Py_DECREF(torch);

    return true;
}

/**
 * @brief Run Python script for model conversion
 */
ConversionResult run_conversion_script(
    const std::string& input_path,
    const std::string& output_path,
    const ConversionConfig& config) {

    ConversionResult result;

    if (!initialize_python()) {
        result.success = false;
        result.error_message = "Failed to initialize Python interpreter";
        return result;
    }

    std::lock_guard<std::mutex> lock(python_mutex);

    // Build Python script
    std::stringstream script;
    script << R"(
import sys
import os

try:
    import torch
    import coremltools as ct
    from ultralytics import YOLO

    # Load model
    model = YOLO(')" << input_path << R"(')

    # Export to CoreML
    export_args = {
        'format': 'coreml',
        'imgsz': [)" << config.input_height << ", " << config.input_width << R"(],
        'nms': )" << (config.include_nms ? "True" : "False") << R"(,
    }
)";

    // Add quantization options
    switch (config.quantization) {
        case QuantizationType::FP16:
            script << "    export_args['half'] = True\n";
            break;
        case QuantizationType::INT8:
            script << "    export_args['int8'] = True\n";
            if (!config.calibration_data_path.empty()) {
                script << "    export_args['data'] = '" << config.calibration_data_path << "'\n";
            }
            break;
        default:
            break;
    }

    script << R"(
    # Run export
    result_path = model.export(**export_args)

    # Move to desired output path
    import shutil
    if os.path.exists(')" << output_path << R"('):
        shutil.rmtree(')" << output_path << R"(')
    shutil.move(result_path, ')" << output_path << R"(')

    print(f"SUCCESS: Model exported to )" << output_path << R"(")

except Exception as e:
    print(f"ERROR: {str(e)}")
    sys.exit(1)
)";

    // Execute script
    PyObject* main_module = PyImport_AddModule("__main__");
    PyObject* global_dict = PyModule_GetDict(main_module);

    PyObject* py_result = PyRun_String(
        script.str().c_str(),
        Py_file_input,
        global_dict,
        global_dict
    );

    if (!py_result) {
        PyErr_Print();
        result.success = false;
        result.error_message = "Python script execution failed";
        return result;
    }

    Py_DECREF(py_result);

    result.success = true;
    result.output_path = output_path;
    result.quantization_applied = to_string(config.quantization);

    return result;
}

/**
 * @brief Run Python script for advanced quantization (W8A8, INT4, etc.)
 */
ConversionResult run_advanced_quantization(
    const std::string& input_path,
    const std::string& output_path,
    const ConversionConfig& config) {

    ConversionResult result;

    if (!initialize_python()) {
        result.success = false;
        result.error_message = "Failed to initialize Python interpreter";
        return result;
    }

    std::lock_guard<std::mutex> lock(python_mutex);

    std::stringstream script;
    script << R"(
import sys
try:
    import coremltools as ct
    from coremltools.optimize.coreml import (
        OpLinearQuantizerConfig,
        OptimizationConfig,
        linear_quantize_weights
    )

    # Load model
    model = ct.models.MLModel(')" << input_path << R"(')

)";

    // Configure quantization based on type
    switch (config.quantization) {
        case QuantizationType::INT8:
            script << R"(
    op_config = OpLinearQuantizerConfig(
        mode="linear_symmetric",
        dtype="int8",
        granularity="per_channel",
        weight_threshold=512
    )
)";
            break;

        case QuantizationType::INT4:
            script << R"(
    op_config = OpLinearQuantizerConfig(
        mode="linear_symmetric",
        dtype="int4",
        granularity="per_block",
        block_size=32
    )
)";
            break;

        case QuantizationType::W8A8:
            script << R"(
    # W8A8 requires PyTorch-level quantization
    # This is a simplified version
    op_config = OpLinearQuantizerConfig(
        mode="linear_symmetric",
        dtype="int8",
        granularity="per_channel"
    )
)";
            break;

        default:
            script << R"(
    op_config = OpLinearQuantizerConfig(dtype="float16")
)";
            break;
    }

    script << R"(
    config = OptimizationConfig(global_config=op_config)
    quantized_model = linear_quantize_weights(model, config=config)
    quantized_model.save(')" << output_path << R"(')

    print("SUCCESS: Quantization complete")

except Exception as e:
    print(f"ERROR: {str(e)}")
    sys.exit(1)
)";

    PyObject* main_module = PyImport_AddModule("__main__");
    PyObject* global_dict = PyModule_GetDict(main_module);

    PyObject* py_result = PyRun_String(
        script.str().c_str(),
        Py_file_input,
        global_dict,
        global_dict
    );

    if (!py_result) {
        PyErr_Print();
        result.success = false;
        result.error_message = "Python quantization script failed";
        return result;
    }

    Py_DECREF(py_result);

    result.success = true;
    result.output_path = output_path;
    result.quantization_applied = to_string(config.quantization);

    return result;
}

} // namespace python_bridge

// Implement Converter methods that require Python
ConversionResult Converter::convert_pytorch(
    const std::string& pt_path,
    const std::string& output_path,
    const ConversionConfig& config) {

    return python_bridge::run_conversion_script(pt_path, output_path, config);
}

ConversionResult Converter::quantize(
    const std::string& input_path,
    const std::string& output_path,
    const ConversionConfig& config) {

    return python_bridge::run_advanced_quantization(input_path, output_path, config);
}

} // namespace yolov12

#endif // YOLOV12_ENABLE_PYTHON
