/**
 * @file convert_cli.cpp
 * @brief Command-line tool for converting YOLO models to CoreML format
 *
 * Usage:
 *   yolov12-convert --input model.onnx --output model.mlpackage --quantize int8
 */

#include <yolov12/yolov12.h>
#include <iostream>
#include <string>
#include <vector>
#include <cstring>

void print_usage(const char* program_name) {
    std::cout << "YOLOv12-Mac Model Converter\n";
    std::cout << "Version: " << yolov12::VERSION << "\n\n";
    std::cout << "Usage: " << program_name << " [options]\n\n";
    std::cout << "Options:\n";
    std::cout << "  -i, --input PATH       Input model path (.pt, .onnx)\n";
    std::cout << "  -o, --output PATH      Output model path (.mlpackage)\n";
    std::cout << "  -q, --quantize TYPE    Quantization type:\n";
    std::cout << "                           none   - FP32 (no quantization)\n";
    std::cout << "                           fp16   - FP16 (default, recommended)\n";
    std::cout << "                           int8   - INT8 linear quantization\n";
    std::cout << "                           int4   - INT4 (requires Python)\n";
    std::cout << "                           w8a8   - INT8 weights+activations (requires Python, M4+)\n";
    std::cout << "  -c, --calibration PATH Path to calibration images (for INT8/W8A8)\n";
    std::cout << "  -n, --samples NUM      Number of calibration samples (default: 128)\n";
    std::cout << "  -W, --width NUM        Input width (default: 640)\n";
    std::cout << "  -H, --height NUM       Input height (default: 640)\n";
    std::cout << "  --nms                  Include NMS in model\n";
    std::cout << "  --overwrite            Overwrite existing output file\n";
    std::cout << "  -v, --verbose          Verbose output\n";
    std::cout << "  -h, --help             Show this help\n";
    std::cout << "\n";
    std::cout << "Examples:\n";
    std::cout << "  # Convert ONNX to CoreML with FP16 (pure C++)\n";
    std::cout << "  " << program_name << " -i yolov12n.onnx -o yolov12n.mlpackage -q fp16\n\n";
    std::cout << "  # Convert PyTorch with INT8 quantization (requires Python)\n";
    std::cout << "  " << program_name << " -i yolov12n.pt -o yolov12n.mlpackage -q int8 -c ./images/\n\n";
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    // Parse arguments
    std::string input_path;
    std::string output_path;
    std::string quantize = "fp16";
    std::string calibration_path;
    int calibration_samples = 128;
    int width = 640;
    int height = 640;
    bool include_nms = false;
    bool overwrite = false;
    bool verbose = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "-i" || arg == "--input") {
            if (i + 1 < argc) input_path = argv[++i];
        } else if (arg == "-o" || arg == "--output") {
            if (i + 1 < argc) output_path = argv[++i];
        } else if (arg == "-q" || arg == "--quantize") {
            if (i + 1 < argc) quantize = argv[++i];
        } else if (arg == "-c" || arg == "--calibration") {
            if (i + 1 < argc) calibration_path = argv[++i];
        } else if (arg == "-n" || arg == "--samples") {
            if (i + 1 < argc) calibration_samples = std::stoi(argv[++i]);
        } else if (arg == "-W" || arg == "--width") {
            if (i + 1 < argc) width = std::stoi(argv[++i]);
        } else if (arg == "-H" || arg == "--height") {
            if (i + 1 < argc) height = std::stoi(argv[++i]);
        } else if (arg == "--nms") {
            include_nms = true;
        } else if (arg == "--overwrite") {
            overwrite = true;
        } else if (arg == "-v" || arg == "--verbose") {
            verbose = true;
        }
    }

    // Validate required arguments
    if (input_path.empty()) {
        std::cerr << "Error: Input path is required (-i)\n";
        return 1;
    }
    if (output_path.empty()) {
        std::cerr << "Error: Output path is required (-o)\n";
        return 1;
    }

    // Build conversion config
    yolov12::ConversionConfig config;
    config.quantization = yolov12::parse_quantization_type(quantize);
    config.calibration_data_path = calibration_path;
    config.calibration_samples = calibration_samples;
    config.input_width = width;
    config.input_height = height;
    config.include_nms = include_nms;
    config.overwrite_existing = overwrite;
    config.verbose = verbose;

    // Check if quantization requires Python
    if (config.requires_python() && !yolov12::is_python_available()) {
        std::cerr << "Error: " << yolov12::to_string(config.quantization)
                  << " quantization requires Python backend.\n";
        std::cerr << "Available quantization types:\n";
        for (auto q : yolov12::get_supported_quantizations()) {
            std::cerr << "  - " << yolov12::to_string(q) << "\n";
        }
        return 1;
    }

    // Print configuration
    if (verbose) {
        std::cout << "Configuration:\n";
        std::cout << "  Input:        " << input_path << "\n";
        std::cout << "  Output:       " << output_path << "\n";
        std::cout << "  Quantization: " << yolov12::to_string(config.quantization) << "\n";
        std::cout << "  Input size:   " << width << "x" << height << "\n";
        std::cout << "  Include NMS:  " << (include_nms ? "yes" : "no") << "\n";
        if (!calibration_path.empty()) {
            std::cout << "  Calibration:  " << calibration_path << " (" << calibration_samples << " samples)\n";
        }
        std::cout << "\n";
    }

    // Run conversion
    std::cout << "Converting model...\n";
    auto result = yolov12::Converter::convert(input_path, output_path, config);

    if (result.success) {
        std::cout << "Conversion successful!\n";
        std::cout << "  Output:       " << result.output_path << "\n";
        std::cout << "  Original:     " << result.original_size_mb << " MB\n";
        std::cout << "  Converted:    " << result.converted_size_mb << " MB\n";
        std::cout << "  Compression:  " << result.compression_ratio << "x\n";
        std::cout << "  Time:         " << result.conversion_time_seconds << " seconds\n";
        return 0;
    } else {
        std::cerr << "Conversion failed: " << result.error_message << "\n";
        return 1;
    }
}
