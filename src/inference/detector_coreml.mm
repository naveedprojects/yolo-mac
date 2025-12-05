#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>
#import <CoreVideo/CoreVideo.h>

#include "yolov12/detector.h"
#include "yolov12/types.h"
#include "yolov12/compute_unit.h"

#include <chrono>
#include <thread>
#include <mutex>

// Forward declarations from other modules
namespace yolov12 {
namespace model_loader {
    void* load_model(const std::string& path, ComputeUnit compute_unit);
    void release_model(void* model_handle);
    ModelInfo get_model_info(void* model_handle, const std::string& path);
    void* run_inference(void* model_handle, void* input_array);
    void release_array(void* array_handle);
    const float* get_array_data(void* array_handle);
    std::vector<int> get_array_shape(void* array_handle);
}

namespace preprocessing {
    MLMultiArray* create_multi_array(const Image& image, int target_width, int target_height);
}

namespace postprocessing {
    std::vector<Detection> process_output(
        const float* output,
        int num_predictions,
        int num_classes,
        float confidence_threshold,
        float iou_threshold,
        int max_detections,
        const std::vector<std::string>& class_names,
        bool is_yolov8_format
    );
}
}

namespace yolov12 {

/**
 * @brief CoreML-based detector implementation
 */
class CoreMLDetector : public Detector {
public:
    CoreMLDetector(const std::string& model_path, const DetectorConfig& config)
        : model_path_(model_path)
        , config_(config)
        , model_handle_(nullptr)
        , is_ready_(false) {

        // Load model
        model_handle_ = model_loader::load_model(model_path, config.compute_unit);

        if (!model_handle_) {
            throw std::runtime_error("Failed to load model: " + model_path);
        }

        // Get model info
        model_info_ = model_loader::get_model_info(model_handle_, model_path);
        model_info_.compute_unit = to_string(config.compute_unit);

        // Set default input size if not detected
        if (model_info_.input_width == 0) {
            model_info_.input_width = 640;
        }
        if (model_info_.input_height == 0) {
            model_info_.input_height = 640;
        }

        // Set default classes if not detected
        if (model_info_.num_classes == 0) {
            model_info_.num_classes = 80;  // COCO default
        }
        if (model_info_.class_names.empty()) {
            model_info_.class_names = get_coco_class_names();
        }

        is_ready_ = true;
    }

    ~CoreMLDetector() override {
        if (model_handle_) {
            model_loader::release_model(model_handle_);
            model_handle_ = nullptr;
        }
    }

    std::vector<Detection> detect(const Image& image) override {
        auto result = detect_with_stats(image);
        return result.detections;
    }

    DetectionResult detect_with_stats(const Image& image) override {
        DetectionResult result;

        if (!is_ready_) {
            result.success = false;
            result.error_message = "Detector not ready";
            return result;
        }

        if (!image.is_valid()) {
            result.success = false;
            result.error_message = "Invalid input image";
            return result;
        }

        try {
            auto total_start = std::chrono::high_resolution_clock::now();

            // Preprocessing
            auto preprocess_start = std::chrono::high_resolution_clock::now();

            MLMultiArray* input_array = preprocessing::create_multi_array(
                image, model_info_.input_width, model_info_.input_height);

            auto preprocess_end = std::chrono::high_resolution_clock::now();
            result.stats.preprocessing_ms = std::chrono::duration<double, std::milli>(
                preprocess_end - preprocess_start).count();

            // Inference
            auto inference_start = std::chrono::high_resolution_clock::now();

            void* output_handle = model_loader::run_inference(
                model_handle_, (__bridge void*)input_array);

            auto inference_end = std::chrono::high_resolution_clock::now();
            result.stats.inference_ms = std::chrono::duration<double, std::milli>(
                inference_end - inference_start).count();

            // Post-processing
            auto postprocess_start = std::chrono::high_resolution_clock::now();

            const float* output_data = model_loader::get_array_data(output_handle);
            std::vector<int> output_shape = model_loader::get_array_shape(output_handle);

            // Determine output format and dimensions
            int num_predictions = 0;
            bool is_yolov8_format = false;

            if (output_shape.size() >= 2) {
                // YOLOv8+ format: [1, 84, 8400] or [84, 8400]
                // vs older format: [1, 8400, 84] or [8400, 84]
                int dim1 = output_shape.size() == 3 ? output_shape[1] : output_shape[0];
                int dim2 = output_shape.size() == 3 ? output_shape[2] : output_shape[1];

                if (dim1 == model_info_.num_classes + 4) {
                    // YOLOv8+ format (transposed)
                    is_yolov8_format = true;
                    num_predictions = dim2;
                } else {
                    // Older format
                    num_predictions = dim1;
                }
            }

            result.detections = postprocessing::process_output(
                output_data,
                num_predictions,
                model_info_.num_classes,
                config_.confidence_threshold,
                config_.iou_threshold,
                config_.max_detections,
                model_info_.class_names,
                is_yolov8_format
            );

            auto postprocess_end = std::chrono::high_resolution_clock::now();
            result.stats.postprocessing_ms = std::chrono::duration<double, std::milli>(
                postprocess_end - postprocess_start).count();

            // Cleanup
            model_loader::release_array(output_handle);

            auto total_end = std::chrono::high_resolution_clock::now();
            result.stats.total_ms = std::chrono::duration<double, std::milli>(
                total_end - total_start).count();

            result.success = true;

        } catch (const std::exception& e) {
            result.success = false;
            result.error_message = e.what();
        }

        return result;
    }

    std::vector<std::vector<Detection>> detect_batch(
        const std::vector<Image>& images) override {

        std::vector<std::vector<Detection>> results;
        results.reserve(images.size());

        // Simple sequential processing for now
        // TODO: Implement true batch processing with MLArrayBatchProvider
        for (const auto& image : images) {
            results.push_back(detect(image));
        }

        return results;
    }

    void detect_async(
        const Image& image,
        std::function<void(DetectionResult)> callback) override {

        // Launch detection in a separate thread
        std::thread([this, &image, callback]() {
            DetectionResult result = detect_with_stats(image);
            callback(result);
        }).detach();
    }

    std::future<DetectionResult> detect_future(const Image& image) override {
        return std::async(std::launch::async, [this, &image]() {
            return detect_with_stats(image);
        });
    }

    ModelInfo get_model_info() const override {
        return model_info_;
    }

    ComputeUnit get_active_compute_unit() const override {
        return config_.compute_unit;
    }

    void get_input_size(int& width, int& height) const override {
        width = model_info_.input_width;
        height = model_info_.input_height;
    }

    void warmup(int iterations) override {
        if (!is_ready_) {
            return;
        }

        // Create dummy image
        std::vector<uint8_t> dummy_data(model_info_.input_width *
                                        model_info_.input_height * 3, 128);
        Image dummy = Image::from_buffer_copy(dummy_data.data(),
                                               model_info_.input_width,
                                               model_info_.input_height,
                                               PixelFormat::RGB);

        for (int i = 0; i < iterations; ++i) {
            detect(dummy);
        }
    }

    bool is_ready() const override {
        return is_ready_;
    }

    void set_confidence_threshold(float threshold) override {
        config_.confidence_threshold = threshold;
    }

    void set_iou_threshold(float threshold) override {
        config_.iou_threshold = threshold;
    }

    void set_max_detections(int max_detections) override {
        config_.max_detections = max_detections;
    }

    float get_confidence_threshold() const override {
        return config_.confidence_threshold;
    }

    float get_iou_threshold() const override {
        return config_.iou_threshold;
    }

private:
    std::string model_path_;
    DetectorConfig config_;
    void* model_handle_;
    ModelInfo model_info_;
    bool is_ready_;
};

// Factory method implementation
std::unique_ptr<Detector> Detector::create_coreml(
    const std::string& model_path,
    const DetectorConfig& config) {

    return std::make_unique<CoreMLDetector>(model_path, config);
}

} // namespace yolov12
