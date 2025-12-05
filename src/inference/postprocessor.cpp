#include "yolov12/types.h"
#include <vector>
#include <algorithm>
#include <cmath>

namespace yolov12 {
namespace postprocessing {

/**
 * @brief Calculate IoU (Intersection over Union) between two boxes
 */
float calculate_iou(const Detection& a, const Detection& b) {
    // Convert from center format to corner format
    float a_x1 = a.x - a.width / 2;
    float a_y1 = a.y - a.height / 2;
    float a_x2 = a.x + a.width / 2;
    float a_y2 = a.y + a.height / 2;

    float b_x1 = b.x - b.width / 2;
    float b_y1 = b.y - b.height / 2;
    float b_x2 = b.x + b.width / 2;
    float b_y2 = b.y + b.height / 2;

    // Calculate intersection
    float inter_x1 = std::max(a_x1, b_x1);
    float inter_y1 = std::max(a_y1, b_y1);
    float inter_x2 = std::min(a_x2, b_x2);
    float inter_y2 = std::min(a_y2, b_y2);

    float inter_width = std::max(0.0f, inter_x2 - inter_x1);
    float inter_height = std::max(0.0f, inter_y2 - inter_y1);
    float inter_area = inter_width * inter_height;

    // Calculate union
    float a_area = a.width * a.height;
    float b_area = b.width * b.height;
    float union_area = a_area + b_area - inter_area;

    if (union_area <= 0) {
        return 0.0f;
    }

    return inter_area / union_area;
}

/**
 * @brief Non-Maximum Suppression
 */
std::vector<Detection> nms(const std::vector<Detection>& detections,
                           float iou_threshold,
                           int max_detections) {
    if (detections.empty()) {
        return {};
    }

    // Sort by confidence (descending)
    std::vector<Detection> sorted_dets = detections;
    std::sort(sorted_dets.begin(), sorted_dets.end(),
              [](const Detection& a, const Detection& b) {
                  return a.confidence > b.confidence;
              });

    std::vector<bool> suppressed(sorted_dets.size(), false);
    std::vector<Detection> results;
    results.reserve(std::min(static_cast<size_t>(max_detections), sorted_dets.size()));

    for (size_t i = 0; i < sorted_dets.size() && results.size() < static_cast<size_t>(max_detections); ++i) {
        if (suppressed[i]) {
            continue;
        }

        results.push_back(sorted_dets[i]);

        // Suppress overlapping boxes of the same class
        for (size_t j = i + 1; j < sorted_dets.size(); ++j) {
            if (suppressed[j]) {
                continue;
            }

            // Only compare boxes of the same class
            if (sorted_dets[i].class_id != sorted_dets[j].class_id) {
                continue;
            }

            float iou = calculate_iou(sorted_dets[i], sorted_dets[j]);
            if (iou > iou_threshold) {
                suppressed[j] = true;
            }
        }
    }

    return results;
}

/**
 * @brief Decode YOLO output tensor to detections
 *
 * YOLO output format: [batch, num_predictions, 4 + num_classes]
 * where 4 = (x, y, w, h) in normalized coordinates
 *
 * @param output Raw model output
 * @param num_predictions Number of prediction boxes
 * @param num_classes Number of classes
 * @param confidence_threshold Minimum confidence
 * @param class_names Vector of class names
 * @return Vector of detections
 */
std::vector<Detection> decode_yolo_output(
    const float* output,
    int num_predictions,
    int num_classes,
    float confidence_threshold,
    const std::vector<std::string>& class_names
) {
    std::vector<Detection> detections;
    detections.reserve(1000);  // Pre-allocate for efficiency

    int stride = 4 + num_classes;  // x, y, w, h + class scores

    for (int i = 0; i < num_predictions; ++i) {
        const float* pred = output + i * stride;

        // Get box coordinates
        float x = pred[0];
        float y = pred[1];
        float w = pred[2];
        float h = pred[3];

        // Find best class
        int best_class = 0;
        float best_score = pred[4];

        for (int c = 1; c < num_classes; ++c) {
            if (pred[4 + c] > best_score) {
                best_score = pred[4 + c];
                best_class = c;
            }
        }

        // Apply confidence threshold
        if (best_score >= confidence_threshold) {
            Detection det;
            det.x = x;
            det.y = y;
            det.width = w;
            det.height = h;
            det.confidence = best_score;
            det.class_id = best_class;

            if (best_class < static_cast<int>(class_names.size())) {
                det.class_name = class_names[best_class];
            } else {
                det.class_name = "class_" + std::to_string(best_class);
            }

            detections.push_back(det);
        }
    }

    return detections;
}

/**
 * @brief Decode YOLOv8/v12 output format (transposed)
 *
 * YOLOv8+ output format: [batch, 4 + num_classes, num_predictions]
 * This is transposed compared to earlier versions.
 */
std::vector<Detection> decode_yolov8_output(
    const float* output,
    int num_predictions,
    int num_classes,
    float confidence_threshold,
    const std::vector<std::string>& class_names
) {
    std::vector<Detection> detections;
    detections.reserve(1000);

    // Output shape: [4 + num_classes, num_predictions]
    // Row 0: x, Row 1: y, Row 2: w, Row 3: h
    // Rows 4+: class scores

    for (int i = 0; i < num_predictions; ++i) {
        // Get box coordinates
        float x = output[0 * num_predictions + i];
        float y = output[1 * num_predictions + i];
        float w = output[2 * num_predictions + i];
        float h = output[3 * num_predictions + i];

        // Find best class
        int best_class = 0;
        float best_score = output[4 * num_predictions + i];

        for (int c = 1; c < num_classes; ++c) {
            float score = output[(4 + c) * num_predictions + i];
            if (score > best_score) {
                best_score = score;
                best_class = c;
            }
        }

        // Apply confidence threshold
        if (best_score >= confidence_threshold) {
            Detection det;
            det.x = x;
            det.y = y;
            det.width = w;
            det.height = h;
            det.confidence = best_score;
            det.class_id = best_class;

            if (best_class < static_cast<int>(class_names.size())) {
                det.class_name = class_names[best_class];
            } else {
                det.class_name = "class_" + std::to_string(best_class);
            }

            detections.push_back(det);
        }
    }

    return detections;
}

/**
 * @brief Scale detections back to original image size
 *
 * Used after letterbox preprocessing to map boxes back to original coordinates.
 */
void scale_detections(
    std::vector<Detection>& detections,
    int model_width,
    int model_height,
    int original_width,
    int original_height,
    int pad_x,
    int pad_y,
    float scale
) {
    for (auto& det : detections) {
        // Remove padding offset (normalized)
        float pad_x_norm = static_cast<float>(pad_x) / model_width;
        float pad_y_norm = static_cast<float>(pad_y) / model_height;

        det.x = (det.x - pad_x_norm) / scale;
        det.y = (det.y - pad_y_norm) / scale;
        det.width = det.width / scale;
        det.height = det.height / scale;

        // Clamp to valid range
        det.x = std::max(0.0f, std::min(1.0f, det.x));
        det.y = std::max(0.0f, std::min(1.0f, det.y));
        det.width = std::max(0.0f, std::min(1.0f - det.x, det.width));
        det.height = std::max(0.0f, std::min(1.0f - det.y, det.height));
    }
}

/**
 * @brief Full post-processing pipeline
 */
std::vector<Detection> process_output(
    const float* output,
    int num_predictions,
    int num_classes,
    float confidence_threshold,
    float iou_threshold,
    int max_detections,
    const std::vector<std::string>& class_names,
    bool is_yolov8_format
) {
    // Decode raw output
    std::vector<Detection> detections;
    if (is_yolov8_format) {
        detections = decode_yolov8_output(output, num_predictions, num_classes,
                                          confidence_threshold, class_names);
    } else {
        detections = decode_yolo_output(output, num_predictions, num_classes,
                                        confidence_threshold, class_names);
    }

    // Apply NMS
    detections = nms(detections, iou_threshold, max_detections);

    return detections;
}

} // namespace postprocessing
} // namespace yolov12
