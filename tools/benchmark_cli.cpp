/**
 * @file benchmark_cli.cpp
 * @brief Command-line tool for benchmarking YOLO model performance
 *
 * Usage:
 *   yolov12-benchmark --model model.mlpackage --compute-unit ane --iterations 100
 */

#include <yolov12/yolov12.h>
#include <iostream>
#include <string>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cmath>

void print_usage(const char* program_name) {
    std::cout << "YOLOv12-Mac Model Benchmark\n";
    std::cout << "Version: " << yolov12::VERSION << "\n\n";
    std::cout << "Usage: " << program_name << " [options]\n\n";
    std::cout << "Options:\n";
    std::cout << "  -m, --model PATH       Model path (.mlpackage, .onnx)\n";
    std::cout << "  -u, --compute-unit UNIT Compute unit:\n";
    std::cout << "                           cpu  - CPU only\n";
    std::cout << "                           gpu  - GPU (Metal)\n";
    std::cout << "                           ane  - Apple Neural Engine\n";
    std::cout << "                           all  - System chooses (default)\n";
    std::cout << "  -i, --iterations NUM   Number of iterations (default: 100)\n";
    std::cout << "  -w, --warmup NUM       Warmup iterations (default: 10)\n";
    std::cout << "  --image PATH           Use actual image instead of dummy data\n";
    std::cout << "  -v, --verbose          Show per-iteration timing\n";
    std::cout << "  -h, --help             Show this help\n";
    std::cout << "\n";
    std::cout << "Examples:\n";
    std::cout << "  # Benchmark on ANE\n";
    std::cout << "  " << program_name << " -m yolov12n.mlpackage -u ane -i 100\n\n";
    std::cout << "  # Compare compute units\n";
    std::cout << "  " << program_name << " -m yolov12n.mlpackage -u cpu -i 50\n";
    std::cout << "  " << program_name << " -m yolov12n.mlpackage -u gpu -i 50\n";
    std::cout << "  " << program_name << " -m yolov12n.mlpackage -u ane -i 50\n\n";
}

struct BenchmarkStats {
    double mean;
    double std_dev;
    double min;
    double max;
    double p50;  // median
    double p95;
    double p99;
};

BenchmarkStats calculate_stats(std::vector<double>& times) {
    BenchmarkStats stats;

    if (times.empty()) {
        return stats;
    }

    // Sort for percentiles
    std::sort(times.begin(), times.end());

    // Mean
    stats.mean = std::accumulate(times.begin(), times.end(), 0.0) / times.size();

    // Standard deviation
    double sq_sum = 0;
    for (auto t : times) {
        sq_sum += (t - stats.mean) * (t - stats.mean);
    }
    stats.std_dev = std::sqrt(sq_sum / times.size());

    // Min/max
    stats.min = times.front();
    stats.max = times.back();

    // Percentiles
    auto percentile = [&times](double p) {
        size_t idx = static_cast<size_t>(p * (times.size() - 1));
        return times[idx];
    };

    stats.p50 = percentile(0.50);
    stats.p95 = percentile(0.95);
    stats.p99 = percentile(0.99);

    return stats;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    // Parse arguments
    std::string model_path;
    std::string compute_unit_str = "all";
    std::string image_path;
    int iterations = 100;
    int warmup = 10;
    bool verbose = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "-m" || arg == "--model") {
            if (i + 1 < argc) model_path = argv[++i];
        } else if (arg == "-u" || arg == "--compute-unit") {
            if (i + 1 < argc) compute_unit_str = argv[++i];
        } else if (arg == "-i" || arg == "--iterations") {
            if (i + 1 < argc) iterations = std::stoi(argv[++i]);
        } else if (arg == "-w" || arg == "--warmup") {
            if (i + 1 < argc) warmup = std::stoi(argv[++i]);
        } else if (arg == "--image") {
            if (i + 1 < argc) image_path = argv[++i];
        } else if (arg == "-v" || arg == "--verbose") {
            verbose = true;
        }
    }

    if (model_path.empty()) {
        std::cerr << "Error: Model path is required (-m)\n";
        return 1;
    }

    // Create detector config
    yolov12::DetectorConfig config;
    config.compute_unit = yolov12::parse_compute_unit(compute_unit_str);

    // Load model
    std::cout << "Loading model: " << model_path << "\n";
    std::cout << "Compute unit:  " << yolov12::to_string(config.compute_unit) << "\n\n";

    std::unique_ptr<yolov12::Detector> detector;
    try {
        detector = yolov12::Detector::create(model_path, config);
    } catch (const std::exception& e) {
        std::cerr << "Failed to load model: " << e.what() << "\n";
        return 1;
    }

    // Get model info
    auto info = detector->get_model_info();
    std::cout << "Model info:\n";
    std::cout << "  Input size:   " << info.input_width << "x" << info.input_height << "\n";
    std::cout << "  Classes:      " << info.num_classes << "\n";
    std::cout << "  Backend:      " << info.backend << "\n";
    std::cout << "\n";

    // Create test image
    yolov12::Image image;
    if (!image_path.empty()) {
        std::cout << "Loading image: " << image_path << "\n";
        try {
            image = yolov12::Image::from_file(image_path);
        } catch (const std::exception& e) {
            std::cerr << "Failed to load image: " << e.what() << "\n";
            return 1;
        }
    } else {
        std::cout << "Using dummy image (" << info.input_width << "x" << info.input_height << ")\n";
        std::vector<uint8_t> dummy_data(info.input_width * info.input_height * 3, 128);
        image = yolov12::Image::from_buffer_copy(
            dummy_data.data(), info.input_width, info.input_height, yolov12::PixelFormat::RGB);
    }

    // Warmup
    std::cout << "\nWarming up (" << warmup << " iterations)...\n";
    detector->warmup(warmup);

    // Benchmark
    std::cout << "Running benchmark (" << iterations << " iterations)...\n\n";

    std::vector<double> total_times;
    std::vector<double> preprocess_times;
    std::vector<double> inference_times;
    std::vector<double> postprocess_times;

    total_times.reserve(iterations);
    preprocess_times.reserve(iterations);
    inference_times.reserve(iterations);
    postprocess_times.reserve(iterations);

    for (int i = 0; i < iterations; ++i) {
        auto result = detector->detect_with_stats(image);

        if (!result.success) {
            std::cerr << "Iteration " << i << " failed: " << result.error_message << "\n";
            continue;
        }

        total_times.push_back(result.stats.total_ms);
        preprocess_times.push_back(result.stats.preprocessing_ms);
        inference_times.push_back(result.stats.inference_ms);
        postprocess_times.push_back(result.stats.postprocessing_ms);

        if (verbose) {
            std::cout << "  [" << i << "] Total: " << result.stats.total_ms << " ms"
                      << " (pre: " << result.stats.preprocessing_ms
                      << ", inf: " << result.stats.inference_ms
                      << ", post: " << result.stats.postprocessing_ms << ")\n";
        }
    }

    // Calculate statistics
    auto total_stats = calculate_stats(total_times);
    auto preprocess_stats = calculate_stats(preprocess_times);
    auto inference_stats = calculate_stats(inference_times);
    auto postprocess_stats = calculate_stats(postprocess_times);

    // Print results
    std::cout << "Results:\n";
    std::cout << "=========================================================\n";
    std::cout << "                    Mean    Std     Min     P50     P95    \n";
    std::cout << "---------------------------------------------------------\n";

    auto print_row = [](const char* name, const BenchmarkStats& s) {
        printf("  %-12s %7.2f %7.2f %7.2f %7.2f %7.2f ms\n",
               name, s.mean, s.std_dev, s.min, s.p50, s.p95);
    };

    print_row("Total", total_stats);
    print_row("Preprocess", preprocess_stats);
    print_row("Inference", inference_stats);
    print_row("Postprocess", postprocess_stats);

    std::cout << "=========================================================\n";
    std::cout << "\n";
    std::cout << "Throughput: " << (1000.0 / total_stats.mean) << " FPS (based on mean)\n";
    std::cout << "            " << (1000.0 / total_stats.p95) << " FPS (based on P95)\n";

    return 0;
}
