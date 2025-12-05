#import <Foundation/Foundation.h>
#import <sys/sysctl.h>

#include "yolov12/yolov12.h"
#include <sstream>
#include <fstream>

namespace yolov12 {

// Global initialization state
static bool g_initialized = false;

bool is_python_available() {
#ifdef YOLOV12_ENABLE_PYTHON
    // In a full implementation, this would actually check Python availability
    return true;
#else
    return false;
#endif
}

std::vector<QuantizationType> get_supported_quantizations() {
    std::vector<QuantizationType> supported = {
        QuantizationType::NONE,
        QuantizationType::FP16,
        QuantizationType::INT8
    };

#ifdef YOLOV12_ENABLE_PYTHON
    supported.push_back(QuantizationType::INT4);
    supported.push_back(QuantizationType::W8A8);
    supported.push_back(QuantizationType::PALETTIZE_4);
    supported.push_back(QuantizationType::PALETTIZE_8);
    supported.push_back(QuantizationType::MIXED);
#endif

    return supported;
}

std::string get_build_info() {
    std::stringstream ss;

    ss << "YOLOv12-Mac v" << VERSION;

#ifdef YOLOV12_ENABLE_COREML
    ss << " +CoreML";
#endif

#ifdef YOLOV12_ENABLE_ONNX
    ss << " +ONNX";
#endif

#ifdef YOLOV12_ENABLE_PYTHON
    ss << " +Python";
#endif

#ifdef YOLOV12_PURE_CPP
    ss << " (Pure C++)";
#endif

#ifdef DEBUG
    ss << " [Debug]";
#else
    ss << " [Release]";
#endif

    return ss.str();
}

void initialize() {
    if (g_initialized) {
        return;
    }

    // Any global initialization can go here
    g_initialized = true;
}

void shutdown() {
    if (!g_initialized) {
        return;
    }

#ifdef YOLOV12_ENABLE_PYTHON
    // Finalize Python if it was initialized
    // python_bridge::finalize_python();
#endif

    g_initialized = false;
}

bool is_apple_silicon() {
    @autoreleasepool {
        // Check CPU brand string
        char buffer[256];
        size_t size = sizeof(buffer);

        if (sysctlbyname("machdep.cpu.brand_string", buffer, &size, nullptr, 0) == 0) {
            std::string brand(buffer);
            return brand.find("Apple") != std::string::npos;
        }

        // Alternative: check for ARM architecture
        int is_arm = 0;
        size = sizeof(is_arm);
        if (sysctlbyname("hw.optional.arm64", &is_arm, &size, nullptr, 0) == 0) {
            return is_arm != 0;
        }

        return false;
    }
}

int get_apple_silicon_generation() {
    if (!is_apple_silicon()) {
        return 0;
    }

    @autoreleasepool {
        char buffer[256];
        size_t size = sizeof(buffer);

        if (sysctlbyname("machdep.cpu.brand_string", buffer, &size, nullptr, 0) == 0) {
            std::string brand(buffer);

            // Parse chip generation from brand string
            // e.g., "Apple M1", "Apple M2 Pro", "Apple M3 Max", "Apple M4"
            if (brand.find("M5") != std::string::npos) return 5;
            if (brand.find("M4") != std::string::npos) return 4;
            if (brand.find("M3") != std::string::npos) return 3;
            if (brand.find("M2") != std::string::npos) return 2;
            if (brand.find("M1") != std::string::npos) return 1;
        }

        // Default to 1 if we know it's Apple Silicon but can't determine generation
        return 1;
    }
}

bool supports_w8a8_acceleration() {
    // W8A8 (INT8 weights + activations) is only hardware-accelerated on M4+ / A17 Pro+
    return get_apple_silicon_generation() >= 4;
}

bool draw_detections(
    const std::string& image_path,
    const std::string& output_path,
    const std::vector<Detection>& detections,
    bool draw_labels) {

    // This would require OpenCV or similar library
    // For now, return false indicating not implemented
    (void)image_path;
    (void)output_path;
    (void)detections;
    (void)draw_labels;
    return false;
}

bool save_detections_json(
    const std::string& output_path,
    const std::vector<Detection>& detections,
    int image_width,
    int image_height) {

    std::ofstream file(output_path);
    if (!file.is_open()) {
        return false;
    }

    file << "{\n";
    file << "  \"image_width\": " << image_width << ",\n";
    file << "  \"image_height\": " << image_height << ",\n";
    file << "  \"detections\": [\n";

    for (size_t i = 0; i < detections.size(); ++i) {
        const auto& det = detections[i];
        auto box = det.to_pixel_coords(image_width, image_height);

        file << "    {\n";
        file << "      \"class_id\": " << det.class_id << ",\n";
        file << "      \"class_name\": \"" << det.class_name << "\",\n";
        file << "      \"confidence\": " << det.confidence << ",\n";
        file << "      \"bbox\": {\n";
        file << "        \"x1\": " << box.x1 << ",\n";
        file << "        \"y1\": " << box.y1 << ",\n";
        file << "        \"x2\": " << box.x2 << ",\n";
        file << "        \"y2\": " << box.y2 << "\n";
        file << "      }\n";
        file << "    }";

        if (i < detections.size() - 1) {
            file << ",";
        }
        file << "\n";
    }

    file << "  ]\n";
    file << "}\n";

    return true;
}

std::string format_detections(const std::vector<Detection>& detections) {
    std::stringstream ss;

    ss << "Detected " << detections.size() << " objects:\n";

    for (size_t i = 0; i < detections.size(); ++i) {
        const auto& det = detections[i];
        ss << "  [" << i << "] " << det.class_name
           << " (" << static_cast<int>(det.confidence * 100) << "%)"
           << " at (" << det.x << ", " << det.y << ")"
           << " size (" << det.width << " x " << det.height << ")\n";
    }

    return ss.str();
}

} // namespace yolov12
