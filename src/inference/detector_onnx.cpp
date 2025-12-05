#ifdef YOLOV12_ENABLE_ONNX

#include "yolov12/detector.h"
#include "yolov12/types.h"
#include <onnxruntime_cxx_api.h>
#include <vector>
#include <chrono>

namespace yolov12 {

// Forward declaration
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

/**
 * @brief ONNX Runtime-based detector with CoreML EP
 */
class ONNXDetector : public Detector {
public:
    ONNXDetector(const std::string& model_path, const DetectorConfig& config)
        : model_path_(model_path)
        , config_(config)
        , env_(ORT_LOGGING_LEVEL_WARNING, "yolov12")
        , is_ready_(false) {

        // Create session options
        Ort::SessionOptions session_options;
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        // Configure CoreML Execution Provider
        std::unordered_map<std::string, std::string> provider_options;

        switch (config.compute_unit) {
            case ComputeUnit::CPU_ONLY:
                provider_options["MLComputeUnits"] = "CPUOnly";
                break;
            case ComputeUnit::ANE_ONLY:
                provider_options["MLComputeUnits"] = "CPUAndNeuralEngine";
                break;
            case ComputeUnit::GPU_ONLY:
            case ComputeUnit::CPU_AND_GPU:
                provider_options["MLComputeUnits"] = "CPUAndGPU";
                break;
            case ComputeUnit::ALL:
            default:
                provider_options["MLComputeUnits"] = "All";
                break;
        }

        // Enable CoreML EP
        try {
            session_options.AppendExecutionProvider("CoreML", provider_options);
        } catch (const Ort::Exception& e) {
            // CoreML EP might not be available, fall back to CPU
            // This is fine, just continue with CPU
        }

        // Create session
        session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), session_options);

        // Get input/output info
        Ort::AllocatorWithDefaultOptions allocator;

        // Input info
        auto input_name = session_->GetInputNameAllocated(0, allocator);
        input_name_ = input_name.get();

        auto input_shape = session_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        if (input_shape.size() >= 4) {
            model_info_.input_height = static_cast<int>(input_shape[2]);
            model_info_.input_width = static_cast<int>(input_shape[3]);
        } else {
            model_info_.input_height = 640;
            model_info_.input_width = 640;
        }

        // Output info
        auto output_name = session_->GetOutputNameAllocated(0, allocator);
        output_name_ = output_name.get();

        auto output_shape = session_->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        if (output_shape.size() >= 2) {
            // Determine num_classes from output shape
            int dim1 = static_cast<int>(output_shape[1]);
            if (dim1 > 4) {
                model_info_.num_classes = dim1 - 4;
            } else {
                model_info_.num_classes = 80;  // Default COCO
            }
        }

        model_info_.path = model_path;
        model_info_.backend = "ONNX Runtime + CoreML EP";
        model_info_.class_names = get_coco_class_names();
        model_info_.compute_unit = to_string(config.compute_unit);

        is_ready_ = true;
    }

    ~ONNXDetector() override = default;

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

        try {
            auto total_start = std::chrono::high_resolution_clock::now();

            // Preprocessing
            auto preprocess_start = std::chrono::high_resolution_clock::now();

            std::vector<float> input_tensor = preprocess(image);

            auto preprocess_end = std::chrono::high_resolution_clock::now();
            result.stats.preprocessing_ms = std::chrono::duration<double, std::milli>(
                preprocess_end - preprocess_start).count();

            // Inference
            auto inference_start = std::chrono::high_resolution_clock::now();

            // Create input tensor
            std::vector<int64_t> input_shape = {1, 3, model_info_.input_height, model_info_.input_width};
            auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

            Ort::Value input_tensor_ort = Ort::Value::CreateTensor<float>(
                memory_info, input_tensor.data(), input_tensor.size(),
                input_shape.data(), input_shape.size());

            // Run inference
            const char* input_names[] = {input_name_.c_str()};
            const char* output_names[] = {output_name_.c_str()};

            auto output_tensors = session_->Run(
                Ort::RunOptions{nullptr},
                input_names, &input_tensor_ort, 1,
                output_names, 1);

            auto inference_end = std::chrono::high_resolution_clock::now();
            result.stats.inference_ms = std::chrono::duration<double, std::milli>(
                inference_end - inference_start).count();

            // Post-processing
            auto postprocess_start = std::chrono::high_resolution_clock::now();

            float* output_data = output_tensors[0].GetTensorMutableData<float>();
            auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();

            int num_predictions = 0;
            bool is_yolov8_format = false;

            if (output_shape.size() >= 2) {
                int dim1 = static_cast<int>(output_shape[1]);
                int dim2 = static_cast<int>(output_shape[2]);

                if (dim1 == model_info_.num_classes + 4) {
                    is_yolov8_format = true;
                    num_predictions = dim2;
                } else {
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

            auto total_end = std::chrono::high_resolution_clock::now();
            result.stats.total_ms = std::chrono::duration<double, std::milli>(
                total_end - total_start).count();

            result.success = true;

        } catch (const Ort::Exception& e) {
            result.success = false;
            result.error_message = e.what();
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

        for (const auto& image : images) {
            results.push_back(detect(image));
        }

        return results;
    }

    void detect_async(
        const Image& image,
        std::function<void(DetectionResult)> callback) override {

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
    std::vector<float> preprocess(const Image& image) {
        int target_w = model_info_.input_width;
        int target_h = model_info_.input_height;

        std::vector<float> tensor(3 * target_h * target_w);

        // Resize and normalize
        float scale_x = static_cast<float>(image.width()) / target_w;
        float scale_y = static_cast<float>(image.height()) / target_h;

        for (int y = 0; y < target_h; ++y) {
            for (int x = 0; x < target_w; ++x) {
                int src_x = static_cast<int>(x * scale_x);
                int src_y = static_cast<int>(y * scale_y);

                src_x = std::min(src_x, image.width() - 1);
                src_y = std::min(src_y, image.height() - 1);

                int src_idx = (src_y * image.width() + src_x) * image.channels();
                int dst_idx = y * target_w + x;

                // CHW format, normalized to [0, 1]
                int r_idx = 0, g_idx = 1, b_idx = 2;
                if (image.format() == PixelFormat::BGR) {
                    r_idx = 2;
                    b_idx = 0;
                }

                tensor[0 * target_h * target_w + dst_idx] = image.data()[src_idx + r_idx] / 255.0f;
                tensor[1 * target_h * target_w + dst_idx] = image.data()[src_idx + g_idx] / 255.0f;
                tensor[2 * target_h * target_w + dst_idx] = image.data()[src_idx + b_idx] / 255.0f;
            }
        }

        return tensor;
    }

    std::string model_path_;
    DetectorConfig config_;
    Ort::Env env_;
    std::unique_ptr<Ort::Session> session_;
    std::string input_name_;
    std::string output_name_;
    ModelInfo model_info_;
    bool is_ready_;
};

std::unique_ptr<Detector> Detector::create_onnx(
    const std::string& model_path,
    const DetectorConfig& config) {

    return std::make_unique<ONNXDetector>(model_path, config);
}

} // namespace yolov12

#endif // YOLOV12_ENABLE_ONNX
