#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>
#import <CoreVideo/CoreVideo.h>
#import <Accelerate/Accelerate.h>
#import <CoreGraphics/CoreGraphics.h>

#include "yolov12/types.h"
#include <vector>
#include <stdexcept>

namespace yolov12 {
namespace preprocessing {

/**
 * @brief Create CVPixelBuffer from Image
 */
CVPixelBufferRef create_pixel_buffer(const Image& image, int target_width, int target_height) {
    if (!image.is_valid()) {
        throw std::runtime_error("Invalid image provided to preprocessor");
    }

    CVPixelBufferRef pixelBuffer = nullptr;

    // Create pixel buffer attributes
    NSDictionary* attributes = @{
        (NSString*)kCVPixelBufferCGImageCompatibilityKey: @YES,
        (NSString*)kCVPixelBufferCGBitmapContextCompatibilityKey: @YES,
        (NSString*)kCVPixelBufferMetalCompatibilityKey: @YES
    };

    // Create pixel buffer
    CVReturn status = CVPixelBufferCreate(
        kCFAllocatorDefault,
        target_width,
        target_height,
        kCVPixelFormatType_32BGRA,  // CoreML prefers BGRA
        (__bridge CFDictionaryRef)attributes,
        &pixelBuffer
    );

    if (status != kCVReturnSuccess) {
        throw std::runtime_error("Failed to create CVPixelBuffer");
    }

    // Lock the buffer for writing
    CVPixelBufferLockBaseAddress(pixelBuffer, 0);

    uint8_t* dest = static_cast<uint8_t*>(CVPixelBufferGetBaseAddress(pixelBuffer));
    size_t destBytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer);

    // Create source vImage buffer
    vImage_Buffer srcBuffer;
    srcBuffer.data = const_cast<uint8_t*>(image.data());
    srcBuffer.width = static_cast<vImagePixelCount>(image.width());
    srcBuffer.height = static_cast<vImagePixelCount>(image.height());
    srcBuffer.rowBytes = image.width() * image.channels();

    // Create destination vImage buffer
    vImage_Buffer destBuffer;
    destBuffer.data = dest;
    destBuffer.width = static_cast<vImagePixelCount>(target_width);
    destBuffer.height = static_cast<vImagePixelCount>(target_height);
    destBuffer.rowBytes = destBytesPerRow;

    // Handle different input formats
    if (image.channels() == 3) {
        // Need to convert RGB to BGRA and resize

        // First, add alpha channel (RGB -> RGBA)
        std::vector<uint8_t> rgbaData(image.width() * image.height() * 4);
        const uint8_t* src = image.data();

        for (int i = 0; i < image.width() * image.height(); ++i) {
            if (image.format() == PixelFormat::RGB) {
                rgbaData[i * 4 + 0] = src[i * 3 + 2];  // B
                rgbaData[i * 4 + 1] = src[i * 3 + 1];  // G
                rgbaData[i * 4 + 2] = src[i * 3 + 0];  // R
                rgbaData[i * 4 + 3] = 255;             // A
            } else {  // BGR
                rgbaData[i * 4 + 0] = src[i * 3 + 0];  // B
                rgbaData[i * 4 + 1] = src[i * 3 + 1];  // G
                rgbaData[i * 4 + 2] = src[i * 3 + 2];  // R
                rgbaData[i * 4 + 3] = 255;             // A
            }
        }

        // Update source buffer
        vImage_Buffer rgbaBuffer;
        rgbaBuffer.data = rgbaData.data();
        rgbaBuffer.width = static_cast<vImagePixelCount>(image.width());
        rgbaBuffer.height = static_cast<vImagePixelCount>(image.height());
        rgbaBuffer.rowBytes = image.width() * 4;

        // Resize using vImage
        vImage_Error error = vImageScale_ARGB8888(&rgbaBuffer, &destBuffer,
                                                   nullptr, kvImageHighQualityResampling);
        if (error != kvImageNoError) {
            CVPixelBufferUnlockBaseAddress(pixelBuffer, 0);
            CVPixelBufferRelease(pixelBuffer);
            throw std::runtime_error("vImage scaling failed");
        }
    } else if (image.channels() == 4) {
        // Already RGBA or BGRA, just need to resize and possibly swap channels

        vImage_Buffer srcBGRA;
        std::vector<uint8_t> bgraData;

        if (image.format() == PixelFormat::RGBA) {
            // Convert RGBA to BGRA
            bgraData.resize(image.width() * image.height() * 4);
            const uint8_t* src = image.data();

            for (int i = 0; i < image.width() * image.height(); ++i) {
                bgraData[i * 4 + 0] = src[i * 4 + 2];  // B
                bgraData[i * 4 + 1] = src[i * 4 + 1];  // G
                bgraData[i * 4 + 2] = src[i * 4 + 0];  // R
                bgraData[i * 4 + 3] = src[i * 4 + 3];  // A
            }

            srcBGRA.data = bgraData.data();
        } else {
            srcBGRA.data = const_cast<uint8_t*>(image.data());
        }

        srcBGRA.width = static_cast<vImagePixelCount>(image.width());
        srcBGRA.height = static_cast<vImagePixelCount>(image.height());
        srcBGRA.rowBytes = image.width() * 4;

        vImage_Error error = vImageScale_ARGB8888(&srcBGRA, &destBuffer,
                                                   nullptr, kvImageHighQualityResampling);
        if (error != kvImageNoError) {
            CVPixelBufferUnlockBaseAddress(pixelBuffer, 0);
            CVPixelBufferRelease(pixelBuffer);
            throw std::runtime_error("vImage scaling failed");
        }
    } else {
        CVPixelBufferUnlockBaseAddress(pixelBuffer, 0);
        CVPixelBufferRelease(pixelBuffer);
        throw std::runtime_error("Unsupported number of channels: " +
                                 std::to_string(image.channels()));
    }

    CVPixelBufferUnlockBaseAddress(pixelBuffer, 0);

    return pixelBuffer;
}

/**
 * @brief Create MLMultiArray from Image for direct tensor input
 *
 * Converts image to normalized float tensor in CHW format (channels first)
 * with values in [0, 1] range.
 */
MLMultiArray* create_multi_array(const Image& image, int target_width, int target_height) {
    if (!image.is_valid()) {
        throw std::runtime_error("Invalid image provided to preprocessor");
    }

    NSError* error = nil;

    // Create MLMultiArray with shape [1, 3, height, width] (NCHW format)
    NSArray<NSNumber*>* shape = @[
        @1,
        @3,
        @(target_height),
        @(target_width)
    ];

    MLMultiArray* multiArray = [[MLMultiArray alloc]
        initWithShape:shape
        dataType:MLMultiArrayDataTypeFloat32
        error:&error];

    if (error) {
        throw std::runtime_error("Failed to create MLMultiArray: " +
                                 std::string([[error localizedDescription] UTF8String]));
    }

    float* dataPtr = (float*)multiArray.dataPointer;

    // Resize image first if needed
    std::vector<uint8_t> resizedData;
    const uint8_t* srcData = image.data();
    int srcWidth = image.width();
    int srcHeight = image.height();

    if (srcWidth != target_width || srcHeight != target_height) {
        // Use simple bilinear interpolation for resizing
        resizedData.resize(target_width * target_height * 3);

        float scaleX = static_cast<float>(srcWidth) / target_width;
        float scaleY = static_cast<float>(srcHeight) / target_height;

        for (int y = 0; y < target_height; ++y) {
            for (int x = 0; x < target_width; ++x) {
                float srcX = x * scaleX;
                float srcY = y * scaleY;

                int x0 = static_cast<int>(srcX);
                int y0 = static_cast<int>(srcY);
                int x1 = std::min(x0 + 1, srcWidth - 1);
                int y1 = std::min(y0 + 1, srcHeight - 1);

                float xFrac = srcX - x0;
                float yFrac = srcY - y0;

                for (int c = 0; c < 3; ++c) {
                    int srcC = c;
                    if (image.format() == PixelFormat::BGR) {
                        srcC = 2 - c;  // Swap R and B
                    }

                    float v00 = srcData[(y0 * srcWidth + x0) * image.channels() + srcC];
                    float v01 = srcData[(y0 * srcWidth + x1) * image.channels() + srcC];
                    float v10 = srcData[(y1 * srcWidth + x0) * image.channels() + srcC];
                    float v11 = srcData[(y1 * srcWidth + x1) * image.channels() + srcC];

                    float value = (1 - xFrac) * (1 - yFrac) * v00 +
                                  xFrac * (1 - yFrac) * v01 +
                                  (1 - xFrac) * yFrac * v10 +
                                  xFrac * yFrac * v11;

                    resizedData[(y * target_width + x) * 3 + c] = static_cast<uint8_t>(value);
                }
            }
        }

        srcData = resizedData.data();
        srcWidth = target_width;
        srcHeight = target_height;
    }

    // Convert to normalized float tensor in CHW format
    // YOLO expects RGB in [0, 1] range
    int planeSize = target_height * target_width;

    for (int y = 0; y < target_height; ++y) {
        for (int x = 0; x < target_width; ++x) {
            int pixelIdx = y * target_width + x;
            int srcIdx = pixelIdx * 3;

            // Handle BGR to RGB conversion if needed
            int rIdx = 0, gIdx = 1, bIdx = 2;
            if (image.format() == PixelFormat::BGR) {
                rIdx = 2;
                bIdx = 0;
            }

            // Write to CHW format: [batch, channel, height, width]
            // Channel 0 = R, Channel 1 = G, Channel 2 = B
            dataPtr[0 * planeSize + pixelIdx] = srcData[srcIdx + rIdx] / 255.0f;  // R
            dataPtr[1 * planeSize + pixelIdx] = srcData[srcIdx + gIdx] / 255.0f;  // G
            dataPtr[2 * planeSize + pixelIdx] = srcData[srcIdx + bIdx] / 255.0f;  // B
        }
    }

    return multiArray;
}

/**
 * @brief Apply letterbox transformation (preserve aspect ratio with padding)
 */
struct LetterboxInfo {
    float scale;
    int pad_x;
    int pad_y;
    int new_width;
    int new_height;
};

LetterboxInfo calculate_letterbox(int src_width, int src_height,
                                  int target_width, int target_height) {
    LetterboxInfo info;

    float scale_w = static_cast<float>(target_width) / src_width;
    float scale_h = static_cast<float>(target_height) / src_height;
    info.scale = std::min(scale_w, scale_h);

    info.new_width = static_cast<int>(src_width * info.scale);
    info.new_height = static_cast<int>(src_height * info.scale);

    info.pad_x = (target_width - info.new_width) / 2;
    info.pad_y = (target_height - info.new_height) / 2;

    return info;
}

} // namespace preprocessing
} // namespace yolov12
