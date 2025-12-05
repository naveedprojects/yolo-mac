#ifndef YOLOV12_TYPES_H
#define YOLOV12_TYPES_H

#include <string>
#include <vector>
#include <cstdint>
#include <memory>

namespace yolov12 {

/**
 * @brief Bounding box detection result
 */
struct Detection {
    int class_id;           ///< Class index (0-79 for COCO)
    float confidence;       ///< Detection confidence [0, 1]
    float x;                ///< Center X coordinate, normalized [0, 1]
    float y;                ///< Center Y coordinate, normalized [0, 1]
    float width;            ///< Box width, normalized [0, 1]
    float height;           ///< Box height, normalized [0, 1]
    std::string class_name; ///< Human-readable class name

    /**
     * @brief Get bounding box in pixel coordinates
     * @param img_width Image width in pixels
     * @param img_height Image height in pixels
     * @return Tuple of (x1, y1, x2, y2) in pixel coordinates
     */
    struct PixelBox {
        int x1, y1, x2, y2;
    };

    PixelBox to_pixel_coords(int img_width, int img_height) const {
        int cx = static_cast<int>(x * img_width);
        int cy = static_cast<int>(y * img_height);
        int w = static_cast<int>(width * img_width);
        int h = static_cast<int>(height * img_height);
        return {
            cx - w / 2,
            cy - h / 2,
            cx + w / 2,
            cy + h / 2
        };
    }
};

/**
 * @brief Image pixel format
 */
enum class PixelFormat {
    RGB,        ///< 3 channels, RGB order
    RGBA,       ///< 4 channels, RGBA order
    BGR,        ///< 3 channels, BGR order (OpenCV default)
    BGRA,       ///< 4 channels, BGRA order
    GRAYSCALE   ///< 1 channel
};

/**
 * @brief Input image wrapper
 */
class Image {
public:
    Image() = default;
    ~Image();

    // Non-copyable, but movable
    Image(const Image&) = delete;
    Image& operator=(const Image&) = delete;
    Image(Image&& other) noexcept;
    Image& operator=(Image&& other) noexcept;

    /**
     * @brief Load image from file
     * @param path Path to image file (supports JPEG, PNG, BMP, etc.)
     * @return Loaded image
     * @throws std::runtime_error if file cannot be loaded
     */
    static Image from_file(const std::string& path);

    /**
     * @brief Create image from raw buffer (does NOT copy data)
     * @param data Pointer to pixel data (must remain valid)
     * @param width Image width in pixels
     * @param height Image height in pixels
     * @param format Pixel format
     * @return Image wrapper
     */
    static Image from_buffer(const uint8_t* data, int width, int height,
                             PixelFormat format = PixelFormat::RGB);

    /**
     * @brief Create image from raw buffer (copies data)
     * @param data Pointer to pixel data
     * @param width Image width in pixels
     * @param height Image height in pixels
     * @param format Pixel format
     * @return Image with owned data
     */
    static Image from_buffer_copy(const uint8_t* data, int width, int height,
                                  PixelFormat format = PixelFormat::RGB);

    // Accessors
    const uint8_t* data() const { return data_; }
    int width() const { return width_; }
    int height() const { return height_; }
    int channels() const { return channels_; }
    PixelFormat format() const { return format_; }
    bool is_valid() const { return data_ != nullptr && width_ > 0 && height_ > 0; }

private:
    const uint8_t* data_ = nullptr;
    int width_ = 0;
    int height_ = 0;
    int channels_ = 0;
    PixelFormat format_ = PixelFormat::RGB;
    bool owns_data_ = false;
    std::unique_ptr<uint8_t[]> owned_buffer_;
};

/**
 * @brief Model metadata information
 */
struct ModelInfo {
    std::string name;                       ///< Model name
    std::string path;                       ///< Model file path
    int input_width = 0;                    ///< Expected input width
    int input_height = 0;                   ///< Expected input height
    int num_classes = 0;                    ///< Number of detection classes
    std::vector<std::string> class_names;   ///< Class name list
    std::string quantization_type;          ///< Quantization applied (FP32, FP16, INT8, etc.)
    std::string backend;                    ///< Backend (CoreML, ONNX)
    std::string compute_unit;               ///< Active compute unit (CPU, GPU, ANE)
    size_t model_size_bytes = 0;            ///< Model file size in bytes
};

/**
 * @brief COCO dataset class names (80 classes)
 */
inline const std::vector<std::string>& get_coco_class_names() {
    static const std::vector<std::string> names = {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
        "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
        "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
        "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
        "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
        "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
        "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
        "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
        "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
        "toothbrush"
    };
    return names;
}

} // namespace yolov12

#endif // YOLOV12_TYPES_H
