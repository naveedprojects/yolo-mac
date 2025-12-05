/**
 * @file basic_detection.cpp
 * @brief Basic example of using YOLOv12-Mac for object detection
 *
 * This example demonstrates:
 * - Loading a CoreML model
 * - Running detection on an image
 * - Printing results
 *
 * Build:
 *   cmake --build build --target basic_detection
 *
 * Run:
 *   ./build/bin/basic_detection model.mlpackage image.jpg
 */

#include <yolov12/yolov12.h>
#include <iostream>
#include <iomanip>

int main(int argc, char* argv[]) {
    // Check arguments
    if (argc < 3) {
        std::cout << "YOLOv12-Mac Basic Detection Example\n\n";
        std::cout << "Usage: " << argv[0] << " <model_path> <image_path> [compute_unit]\n\n";
        std::cout << "Arguments:\n";
        std::cout << "  model_path   Path to .mlpackage or .onnx model\n";
        std::cout << "  image_path   Path to input image\n";
        std::cout << "  compute_unit Optional: cpu, gpu, ane, or all (default: all)\n\n";
        std::cout << "Example:\n";
        std::cout << "  " << argv[0] << " yolov12n.mlpackage test.jpg ane\n";
        return 1;
    }

    std::string model_path = argv[1];
    std::string image_path = argv[2];
    std::string compute_unit_str = (argc > 3) ? argv[3] : "all";

    // Initialize library (optional - happens automatically)
    yolov12::initialize();

    // Print library info
    std::cout << "YOLOv12-Mac v" << yolov12::VERSION << "\n";
    std::cout << "Build: " << yolov12::get_build_info() << "\n";

    if (yolov12::is_apple_silicon()) {
        std::cout << "Running on Apple Silicon M" << yolov12::get_apple_silicon_generation() << "\n";
        if (yolov12::supports_w8a8_acceleration()) {
            std::cout << "W8A8 acceleration: supported\n";
        }
    }
    std::cout << "\n";

    // Configure detector
    yolov12::DetectorConfig config;
    config.compute_unit = yolov12::parse_compute_unit(compute_unit_str);
    config.confidence_threshold = 0.5f;  // Only show detections with >50% confidence
    config.iou_threshold = 0.45f;
    config.max_detections = 100;

    std::cout << "Loading model: " << model_path << "\n";
    std::cout << "Compute unit:  " << yolov12::to_string(config.compute_unit) << "\n\n";

    // Create detector
    std::unique_ptr<yolov12::Detector> detector;
    try {
        detector = yolov12::Detector::create(model_path, config);
    } catch (const std::exception& e) {
        std::cerr << "Error loading model: " << e.what() << "\n";
        return 1;
    }

    // Print model info
    auto model_info = detector->get_model_info();
    std::cout << "Model info:\n";
    std::cout << "  Name:        " << model_info.name << "\n";
    std::cout << "  Input size:  " << model_info.input_width << "x" << model_info.input_height << "\n";
    std::cout << "  Classes:     " << model_info.num_classes << "\n";
    std::cout << "  Backend:     " << model_info.backend << "\n";
    std::cout << "  Size:        " << (model_info.model_size_bytes / (1024.0 * 1024.0)) << " MB\n\n";

    // Warmup (first inference is slower)
    std::cout << "Warming up...\n";
    detector->warmup(3);

    // Load image
    std::cout << "Loading image: " << image_path << "\n";
    yolov12::Image image;
    try {
        image = yolov12::Image::from_file(image_path);
    } catch (const std::exception& e) {
        std::cerr << "Error loading image: " << e.what() << "\n";
        return 1;
    }
    std::cout << "Image size: " << image.width() << "x" << image.height() << "\n\n";

    // Run detection
    std::cout << "Running detection...\n";
    auto result = detector->detect_with_stats(image);

    if (!result.success) {
        std::cerr << "Detection failed: " << result.error_message << "\n";
        return 1;
    }

    // Print timing
    std::cout << "\nTiming:\n";
    std::cout << "  Preprocessing:  " << std::fixed << std::setprecision(2)
              << result.stats.preprocessing_ms << " ms\n";
    std::cout << "  Inference:      " << result.stats.inference_ms << " ms\n";
    std::cout << "  Postprocessing: " << result.stats.postprocessing_ms << " ms\n";
    std::cout << "  Total:          " << result.stats.total_ms << " ms\n";
    std::cout << "  FPS:            " << result.stats.fps() << "\n\n";

    // Print detections
    std::cout << "Detections: " << result.detections.size() << "\n";
    std::cout << "-------------------------------------------\n";

    for (size_t i = 0; i < result.detections.size(); ++i) {
        const auto& det = result.detections[i];

        // Get pixel coordinates
        auto box = det.to_pixel_coords(image.width(), image.height());

        std::cout << "[" << i << "] " << det.class_name
                  << " (" << std::fixed << std::setprecision(1) << (det.confidence * 100) << "%)"
                  << " at (" << box.x1 << ", " << box.y1 << ") - ("
                  << box.x2 << ", " << box.y2 << ")\n";
    }

    std::cout << "-------------------------------------------\n";

    // Format detections as string (utility function)
    // std::cout << "\nFormatted output:\n";
    // std::cout << yolov12::format_detections(result.detections) << "\n";

    // Cleanup (optional - happens automatically)
    yolov12::shutdown();

    return 0;
}
