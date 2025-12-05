/**
 * @file batch_detection.cpp
 * @brief Example of batch detection for processing multiple images
 *
 * This example demonstrates:
 * - Processing multiple images efficiently
 * - Using async detection
 * - Comparing batch vs sequential performance
 *
 * Build:
 *   cmake --build build --target batch_detection
 *
 * Run:
 *   ./build/bin/batch_detection model.mlpackage image1.jpg image2.jpg image3.jpg
 */

#include <yolov12/yolov12.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <future>

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cout << "YOLOv12-Mac Batch Detection Example\n\n";
        std::cout << "Usage: " << argv[0] << " <model_path> <image1> [image2] [image3] ...\n\n";
        std::cout << "Example:\n";
        std::cout << "  " << argv[0] << " yolov12n.mlpackage img1.jpg img2.jpg img3.jpg\n";
        return 1;
    }

    std::string model_path = argv[1];
    std::vector<std::string> image_paths;
    for (int i = 2; i < argc; ++i) {
        image_paths.push_back(argv[i]);
    }

    std::cout << "YOLOv12-Mac Batch Detection\n";
    std::cout << "Model: " << model_path << "\n";
    std::cout << "Images: " << image_paths.size() << "\n\n";

    // Create detector with ANE preference
    yolov12::DetectorConfig config;
    config.compute_unit = yolov12::ComputeUnit::ALL;
    config.confidence_threshold = 0.5f;

    std::unique_ptr<yolov12::Detector> detector;
    try {
        detector = yolov12::Detector::create(model_path, config);
    } catch (const std::exception& e) {
        std::cerr << "Error loading model: " << e.what() << "\n";
        return 1;
    }

    // Warmup
    detector->warmup(3);

    // Load all images
    std::cout << "Loading images...\n";
    std::vector<yolov12::Image> images;
    for (const auto& path : image_paths) {
        try {
            images.push_back(yolov12::Image::from_file(path));
            std::cout << "  Loaded: " << path << " ("
                      << images.back().width() << "x"
                      << images.back().height() << ")\n";
        } catch (const std::exception& e) {
            std::cerr << "  Failed to load " << path << ": " << e.what() << "\n";
        }
    }
    std::cout << "\n";

    if (images.empty()) {
        std::cerr << "No images loaded.\n";
        return 1;
    }

    // =========== Method 1: Sequential processing ===========
    std::cout << "Method 1: Sequential processing\n";
    std::cout << "--------------------------------\n";

    auto seq_start = std::chrono::high_resolution_clock::now();

    std::vector<std::vector<yolov12::Detection>> seq_results;
    for (size_t i = 0; i < images.size(); ++i) {
        auto detections = detector->detect(images[i]);
        seq_results.push_back(detections);
        std::cout << "  Image " << i << ": " << detections.size() << " detections\n";
    }

    auto seq_end = std::chrono::high_resolution_clock::now();
    double seq_time = std::chrono::duration<double, std::milli>(seq_end - seq_start).count();

    std::cout << "Total time: " << seq_time << " ms\n";
    std::cout << "Per image:  " << (seq_time / images.size()) << " ms\n\n";

    // =========== Method 2: Batch processing ===========
    std::cout << "Method 2: Batch processing\n";
    std::cout << "--------------------------\n";

    auto batch_start = std::chrono::high_resolution_clock::now();

    auto batch_results = detector->detect_batch(images);

    auto batch_end = std::chrono::high_resolution_clock::now();
    double batch_time = std::chrono::duration<double, std::milli>(batch_end - batch_start).count();

    for (size_t i = 0; i < batch_results.size(); ++i) {
        std::cout << "  Image " << i << ": " << batch_results[i].size() << " detections\n";
    }

    std::cout << "Total time: " << batch_time << " ms\n";
    std::cout << "Per image:  " << (batch_time / images.size()) << " ms\n\n";

    // =========== Method 3: Async with futures ===========
    std::cout << "Method 3: Async with futures\n";
    std::cout << "----------------------------\n";

    auto async_start = std::chrono::high_resolution_clock::now();

    // Launch all detections asynchronously
    std::vector<std::future<yolov12::DetectionResult>> futures;
    for (size_t i = 0; i < images.size(); ++i) {
        futures.push_back(detector->detect_future(images[i]));
    }

    // Collect results
    std::vector<std::vector<yolov12::Detection>> async_results;
    for (size_t i = 0; i < futures.size(); ++i) {
        auto result = futures[i].get();
        async_results.push_back(result.detections);
        std::cout << "  Image " << i << ": " << result.detections.size()
                  << " detections (inference: " << result.stats.inference_ms << " ms)\n";
    }

    auto async_end = std::chrono::high_resolution_clock::now();
    double async_time = std::chrono::duration<double, std::milli>(async_end - async_start).count();

    std::cout << "Total time: " << async_time << " ms\n";
    std::cout << "Per image:  " << (async_time / images.size()) << " ms\n\n";

    // =========== Summary ===========
    std::cout << "Summary\n";
    std::cout << "=======\n";
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Sequential: " << seq_time << " ms total, "
              << (seq_time / images.size()) << " ms/image\n";
    std::cout << "Batch:      " << batch_time << " ms total, "
              << (batch_time / images.size()) << " ms/image\n";
    std::cout << "Async:      " << async_time << " ms total, "
              << (async_time / images.size()) << " ms/image\n\n";

    // Print detailed results for first image
    std::cout << "Detailed results for first image:\n";
    std::cout << "---------------------------------\n";
    for (const auto& det : seq_results[0]) {
        std::cout << "  " << det.class_name << ": "
                  << (det.confidence * 100) << "%\n";
    }

    return 0;
}
