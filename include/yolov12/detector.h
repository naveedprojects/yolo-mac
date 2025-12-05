#ifndef YOLOV12_DETECTOR_H
#define YOLOV12_DETECTOR_H

#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <future>
#include "types.h"
#include "compute_unit.h"

namespace yolov12 {

/**
 * @brief Configuration for detector initialization
 */
struct DetectorConfig {
    ComputeUnit compute_unit = ComputeUnit::ALL;    ///< Preferred compute unit
    float confidence_threshold = 0.25f;              ///< Min confidence for detection
    float iou_threshold = 0.45f;                     ///< IoU threshold for NMS
    int max_detections = 100;                        ///< Maximum detections to return
    bool async_inference = false;                    ///< Enable async inference
    int num_threads = 0;                             ///< CPU threads (0 = auto)

    /**
     * @brief Create config for ANE-optimized inference
     */
    static DetectorConfig for_ane() {
        DetectorConfig config;
        config.compute_unit = ComputeUnit::ANE_ONLY;
        return config;
    }

    /**
     * @brief Create config for GPU inference
     */
    static DetectorConfig for_gpu() {
        DetectorConfig config;
        config.compute_unit = ComputeUnit::GPU_ONLY;
        return config;
    }

    /**
     * @brief Create config for CPU-only inference
     */
    static DetectorConfig for_cpu() {
        DetectorConfig config;
        config.compute_unit = ComputeUnit::CPU_ONLY;
        return config;
    }
};

/**
 * @brief Inference timing statistics
 */
struct InferenceStats {
    double preprocessing_ms = 0.0;   ///< Image preprocessing time
    double inference_ms = 0.0;       ///< Model inference time
    double postprocessing_ms = 0.0;  ///< NMS and box decoding time
    double total_ms = 0.0;           ///< Total time

    double fps() const {
        return total_ms > 0 ? 1000.0 / total_ms : 0.0;
    }
};

/**
 * @brief Detection result with optional timing
 */
struct DetectionResult {
    std::vector<Detection> detections;
    InferenceStats stats;
    bool success = true;
    std::string error_message;
};

/**
 * @brief Abstract detector interface
 *
 * Factory method `create()` returns appropriate implementation based on
 * model file type (CoreML or ONNX).
 */
class Detector {
public:
    virtual ~Detector() = default;

    // Disable copy
    Detector(const Detector&) = delete;
    Detector& operator=(const Detector&) = delete;

    /**
     * @brief Create detector from model file
     *
     * Automatically detects model type from file extension:
     * - .mlpackage, .mlmodel → CoreML backend
     * - .onnx → ONNX Runtime backend
     *
     * @param model_path Path to model file
     * @param config Detector configuration
     * @return Unique pointer to detector
     * @throws std::runtime_error if model cannot be loaded
     */
    static std::unique_ptr<Detector> create(
        const std::string& model_path,
        const DetectorConfig& config = {}
    );

    /**
     * @brief Create CoreML detector explicitly
     * @param model_path Path to .mlpackage or .mlmodel
     * @param config Detector configuration
     * @return Unique pointer to detector
     */
    static std::unique_ptr<Detector> create_coreml(
        const std::string& model_path,
        const DetectorConfig& config = {}
    );

    /**
     * @brief Create ONNX Runtime detector explicitly
     * @param model_path Path to .onnx file
     * @param config Detector configuration
     * @return Unique pointer to detector
     */
    static std::unique_ptr<Detector> create_onnx(
        const std::string& model_path,
        const DetectorConfig& config = {}
    );

    // ==================== Inference Methods ====================

    /**
     * @brief Detect objects in image
     * @param image Input image
     * @return Vector of detections
     */
    virtual std::vector<Detection> detect(const Image& image) = 0;

    /**
     * @brief Detect objects with timing statistics
     * @param image Input image
     * @return Detection result with stats
     */
    virtual DetectionResult detect_with_stats(const Image& image) = 0;

    /**
     * @brief Batch detection for multiple images
     *
     * More efficient than calling detect() multiple times.
     *
     * @param images Vector of input images
     * @return Vector of detection vectors (one per image)
     */
    virtual std::vector<std::vector<Detection>> detect_batch(
        const std::vector<Image>& images
    ) = 0;

    /**
     * @brief Asynchronous detection with callback
     * @param image Input image
     * @param callback Function called with results
     */
    virtual void detect_async(
        const Image& image,
        std::function<void(DetectionResult)> callback
    ) = 0;

    /**
     * @brief Asynchronous detection returning future
     * @param image Input image
     * @return Future that will contain results
     */
    virtual std::future<DetectionResult> detect_future(const Image& image) = 0;

    // ==================== Model Info ====================

    /**
     * @brief Get model metadata
     * @return Model information
     */
    virtual ModelInfo get_model_info() const = 0;

    /**
     * @brief Get active compute unit
     *
     * May differ from requested unit if not available.
     *
     * @return Currently active compute unit
     */
    virtual ComputeUnit get_active_compute_unit() const = 0;

    /**
     * @brief Get expected input size
     * @param width Output: expected width
     * @param height Output: expected height
     */
    virtual void get_input_size(int& width, int& height) const = 0;

    // ==================== Lifecycle ====================

    /**
     * @brief Warm up model for consistent performance
     *
     * First inference is typically slower. Call this to ensure
     * consistent timing in benchmarks.
     *
     * @param iterations Number of warmup iterations (default: 3)
     */
    virtual void warmup(int iterations = 3) = 0;

    /**
     * @brief Check if model is loaded and ready
     * @return true if ready for inference
     */
    virtual bool is_ready() const = 0;

    // ==================== Configuration ====================

    /**
     * @brief Set confidence threshold
     * @param threshold New threshold [0, 1]
     */
    virtual void set_confidence_threshold(float threshold) = 0;

    /**
     * @brief Set IoU threshold for NMS
     * @param threshold New threshold [0, 1]
     */
    virtual void set_iou_threshold(float threshold) = 0;

    /**
     * @brief Set maximum detections
     * @param max_detections Maximum number to return
     */
    virtual void set_max_detections(int max_detections) = 0;

    /**
     * @brief Get current confidence threshold
     */
    virtual float get_confidence_threshold() const = 0;

    /**
     * @brief Get current IoU threshold
     */
    virtual float get_iou_threshold() const = 0;

protected:
    Detector() = default;
};

} // namespace yolov12

#endif // YOLOV12_DETECTOR_H
