#include "yolov12/types.h"
#include <stdexcept>
#include <cstring>

// Include stb_image for image loading
#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"

namespace yolov12 {

// ==================== Image Implementation ====================

Image::~Image() {
    // owned_buffer_ is automatically cleaned up by unique_ptr
}

Image::Image(Image&& other) noexcept
    : data_(other.data_)
    , width_(other.width_)
    , height_(other.height_)
    , channels_(other.channels_)
    , format_(other.format_)
    , owns_data_(other.owns_data_)
    , owned_buffer_(std::move(other.owned_buffer_)) {
    other.data_ = nullptr;
    other.width_ = 0;
    other.height_ = 0;
    other.channels_ = 0;
    other.owns_data_ = false;
}

Image& Image::operator=(Image&& other) noexcept {
    if (this != &other) {
        data_ = other.data_;
        width_ = other.width_;
        height_ = other.height_;
        channels_ = other.channels_;
        format_ = other.format_;
        owns_data_ = other.owns_data_;
        owned_buffer_ = std::move(other.owned_buffer_);

        other.data_ = nullptr;
        other.width_ = 0;
        other.height_ = 0;
        other.channels_ = 0;
        other.owns_data_ = false;
    }
    return *this;
}

Image Image::from_file(const std::string& path) {
    int width, height, channels;

    // Load image using stb_image
    unsigned char* data = stbi_load(path.c_str(), &width, &height, &channels, 0);

    if (!data) {
        throw std::runtime_error("Failed to load image: " + path +
                                 " (" + stbi_failure_reason() + ")");
    }

    Image image;
    image.width_ = width;
    image.height_ = height;
    image.channels_ = channels;
    image.owns_data_ = true;

    // Determine format
    switch (channels) {
        case 1:
            image.format_ = PixelFormat::GRAYSCALE;
            break;
        case 3:
            image.format_ = PixelFormat::RGB;
            break;
        case 4:
            image.format_ = PixelFormat::RGBA;
            break;
        default:
            stbi_image_free(data);
            throw std::runtime_error("Unsupported number of channels: " +
                                     std::to_string(channels));
    }

    // Transfer ownership to unique_ptr
    size_t size = static_cast<size_t>(width * height * channels);
    image.owned_buffer_ = std::make_unique<uint8_t[]>(size);
    std::memcpy(image.owned_buffer_.get(), data, size);
    image.data_ = image.owned_buffer_.get();

    // Free stb_image's allocation
    stbi_image_free(data);

    return image;
}

Image Image::from_buffer(const uint8_t* data, int width, int height,
                         PixelFormat format) {
    Image image;
    image.data_ = data;
    image.width_ = width;
    image.height_ = height;
    image.owns_data_ = false;
    image.format_ = format;

    switch (format) {
        case PixelFormat::GRAYSCALE:
            image.channels_ = 1;
            break;
        case PixelFormat::RGB:
        case PixelFormat::BGR:
            image.channels_ = 3;
            break;
        case PixelFormat::RGBA:
        case PixelFormat::BGRA:
            image.channels_ = 4;
            break;
    }

    return image;
}

Image Image::from_buffer_copy(const uint8_t* data, int width, int height,
                              PixelFormat format) {
    Image image;
    image.width_ = width;
    image.height_ = height;
    image.format_ = format;
    image.owns_data_ = true;

    switch (format) {
        case PixelFormat::GRAYSCALE:
            image.channels_ = 1;
            break;
        case PixelFormat::RGB:
        case PixelFormat::BGR:
            image.channels_ = 3;
            break;
        case PixelFormat::RGBA:
        case PixelFormat::BGRA:
            image.channels_ = 4;
            break;
    }

    size_t size = static_cast<size_t>(width * height * image.channels_);
    image.owned_buffer_ = std::make_unique<uint8_t[]>(size);
    std::memcpy(image.owned_buffer_.get(), data, size);
    image.data_ = image.owned_buffer_.get();

    return image;
}

} // namespace yolov12
