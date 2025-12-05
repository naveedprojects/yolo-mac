#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>

#include "yolov12/types.h"
#include "yolov12/compute_unit.h"
#include <string>
#include <stdexcept>

namespace yolov12 {
namespace model_loader {

/**
 * @brief Convert ComputeUnit to MLComputeUnits
 */
MLComputeUnits to_ml_compute_units(ComputeUnit unit) {
    switch (unit) {
        case ComputeUnit::CPU_ONLY:
            return MLComputeUnitsCPUOnly;
        case ComputeUnit::GPU_ONLY:
            // Note: There's no GPU-only option, use CPU+GPU
            return MLComputeUnitsCPUAndGPU;
        case ComputeUnit::ANE_ONLY:
            // Use CPU+NeuralEngine (no ANE-only option)
            return MLComputeUnitsCPUAndNeuralEngine;
        case ComputeUnit::CPU_AND_GPU:
            return MLComputeUnitsCPUAndGPU;
        case ComputeUnit::CPU_AND_ANE:
            return MLComputeUnitsCPUAndNeuralEngine;
        case ComputeUnit::ALL:
        default:
            return MLComputeUnitsAll;
    }
}

/**
 * @brief Get actual compute unit string from MLModel
 */
std::string get_compute_unit_string(MLComputeUnits units) {
    switch (units) {
        case MLComputeUnitsCPUOnly:
            return "CPU";
        case MLComputeUnitsCPUAndGPU:
            return "CPU+GPU";
        case MLComputeUnitsCPUAndNeuralEngine:
            return "CPU+ANE";
        case MLComputeUnitsAll:
            return "ALL";
        default:
            return "Unknown";
    }
}

/**
 * @brief Load CoreML model from path
 *
 * @param path Path to .mlpackage or .mlmodel file
 * @param compute_unit Desired compute unit
 * @return Retained MLModel pointer (caller must release)
 */
void* load_model(const std::string& path, ComputeUnit compute_unit) {
    @autoreleasepool {
        NSError* error = nil;

        // Create URL from path
        NSString* nsPath = [NSString stringWithUTF8String:path.c_str()];
        NSURL* modelURL = [NSURL fileURLWithPath:nsPath];

        // Check if file exists
        if (![[NSFileManager defaultManager] fileExistsAtPath:nsPath]) {
            throw std::runtime_error("Model file not found: " + path);
        }

        // Create configuration
        MLModelConfiguration* config = [[MLModelConfiguration alloc] init];
        config.computeUnits = to_ml_compute_units(compute_unit);

        // For better performance on Apple Silicon
        if (@available(macOS 14.0, *)) {
            // Enable optimizations available in macOS 14+
            config.allowLowPrecisionAccumulationOnGPU = YES;
        }

        // Load compiled model
        MLModel* model = nil;

        // Check if it's a .mlpackage (needs compilation) or .mlmodelc (pre-compiled)
        if ([nsPath hasSuffix:@".mlpackage"] || [nsPath hasSuffix:@".mlmodel"]) {
            // Compile the model first
            NSURL* compiledURL = [MLModel compileModelAtURL:modelURL error:&error];

            if (error) {
                throw std::runtime_error("Failed to compile model: " +
                    std::string([[error localizedDescription] UTF8String]));
            }

            model = [MLModel modelWithContentsOfURL:compiledURL
                                      configuration:config
                                              error:&error];

            // Clean up compiled model (it's in a temp directory)
            // Note: In production, you might want to cache this
        } else if ([nsPath hasSuffix:@".mlmodelc"]) {
            // Already compiled
            model = [MLModel modelWithContentsOfURL:modelURL
                                      configuration:config
                                              error:&error];
        } else {
            throw std::runtime_error("Unsupported model format. Use .mlpackage, .mlmodel, or .mlmodelc");
        }

        if (error || !model) {
            throw std::runtime_error("Failed to load model: " +
                (error ? std::string([[error localizedDescription] UTF8String]) : "Unknown error"));
        }

        // Retain and return as void* for C++ storage
        return (__bridge_retained void*)model;
    }
}

/**
 * @brief Release a loaded model
 */
void release_model(void* model_handle) {
    if (model_handle) {
        MLModel* model = (__bridge_transfer MLModel*)model_handle;
        model = nil;  // Release
    }
}

/**
 * @brief Get model description
 */
ModelInfo get_model_info(void* model_handle, const std::string& path) {
    ModelInfo info;
    info.path = path;
    info.backend = "CoreML";

    if (!model_handle) {
        return info;
    }

    @autoreleasepool {
        MLModel* model = (__bridge MLModel*)model_handle;
        MLModelDescription* desc = model.modelDescription;

        // Get model metadata
        NSDictionary* metadata = desc.metadata;
        if (metadata) {
            NSString* author = metadata[MLModelAuthorKey];
            NSString* description = metadata[MLModelDescriptionKey];

            if (author) {
                info.name = std::string([author UTF8String]);
            }
        }

        // Get input description
        NSDictionary<NSString*, MLFeatureDescription*>* inputs = desc.inputDescriptionsByName;
        for (NSString* key in inputs) {
            MLFeatureDescription* inputDesc = inputs[key];

            if (inputDesc.type == MLFeatureTypeMultiArray) {
                MLMultiArrayConstraint* constraint = inputDesc.multiArrayConstraint;
                NSArray<NSNumber*>* shape = constraint.shape;

                if (shape.count >= 4) {
                    // Assume NCHW format
                    info.input_height = [shape[2] intValue];
                    info.input_width = [shape[3] intValue];
                }
            } else if (inputDesc.type == MLFeatureTypeImage) {
                MLImageConstraint* constraint = inputDesc.imageConstraint;
                info.input_width = static_cast<int>(constraint.pixelsWide);
                info.input_height = static_cast<int>(constraint.pixelsHigh);
            }
        }

        // Get output description for num_classes
        NSDictionary<NSString*, MLFeatureDescription*>* outputs = desc.outputDescriptionsByName;
        for (NSString* key in outputs) {
            MLFeatureDescription* outputDesc = outputs[key];

            if (outputDesc.type == MLFeatureTypeMultiArray) {
                MLMultiArrayConstraint* constraint = outputDesc.multiArrayConstraint;
                NSArray<NSNumber*>* shape = constraint.shape;

                // YOLOv8+ format: [1, 84, 8400] for COCO (84 = 4 + 80 classes)
                if (shape.count >= 2) {
                    int dim1 = [shape[1] intValue];
                    if (dim1 > 4) {
                        info.num_classes = dim1 - 4;  // Subtract box coordinates
                    }
                }
            }
        }

        // Set default class names if COCO
        if (info.num_classes == 80) {
            info.class_names = get_coco_class_names();
        }

        // Get file size
        NSString* nsPath = [NSString stringWithUTF8String:path.c_str()];
        NSDictionary* attrs = [[NSFileManager defaultManager] attributesOfItemAtPath:nsPath error:nil];
        if (attrs) {
            info.model_size_bytes = [attrs fileSize];
        }
    }

    return info;
}

/**
 * @brief Run inference on the model
 *
 * @param model_handle Loaded model handle
 * @param input_array MLMultiArray input
 * @return Output MLMultiArray (retained, caller must release)
 */
void* run_inference(void* model_handle, void* input_array) {
    if (!model_handle || !input_array) {
        throw std::runtime_error("Invalid model or input");
    }

    @autoreleasepool {
        MLModel* model = (__bridge MLModel*)model_handle;
        MLMultiArray* input = (__bridge MLMultiArray*)input_array;

        NSError* error = nil;

        // Get input feature name
        MLModelDescription* desc = model.modelDescription;
        NSString* inputName = [[desc.inputDescriptionsByName allKeys] firstObject];

        if (!inputName) {
            throw std::runtime_error("Could not determine input feature name");
        }

        // Create input feature provider
        MLDictionaryFeatureProvider* inputProvider =
            [[MLDictionaryFeatureProvider alloc]
                initWithDictionary:@{inputName: input}
                             error:&error];

        if (error) {
            throw std::runtime_error("Failed to create input provider: " +
                std::string([[error localizedDescription] UTF8String]));
        }

        // Run prediction
        id<MLFeatureProvider> output = [model predictionFromFeatures:inputProvider
                                                               error:&error];

        if (error) {
            throw std::runtime_error("Inference failed: " +
                std::string([[error localizedDescription] UTF8String]));
        }

        // Get output feature name
        NSString* outputName = [[desc.outputDescriptionsByName allKeys] firstObject];

        if (!outputName) {
            throw std::runtime_error("Could not determine output feature name");
        }

        // Get output array
        MLFeatureValue* outputValue = [output featureValueForName:outputName];
        MLMultiArray* outputArray = outputValue.multiArrayValue;

        if (!outputArray) {
            throw std::runtime_error("Output is not a MultiArray");
        }

        // Retain and return
        return (__bridge_retained void*)outputArray;
    }
}

/**
 * @brief Run inference with CVPixelBuffer input
 */
void* run_inference_pixelbuffer(void* model_handle, CVPixelBufferRef pixel_buffer) {
    if (!model_handle || !pixel_buffer) {
        throw std::runtime_error("Invalid model or input");
    }

    @autoreleasepool {
        MLModel* model = (__bridge MLModel*)model_handle;

        NSError* error = nil;

        // Get input feature name
        MLModelDescription* desc = model.modelDescription;
        NSString* inputName = [[desc.inputDescriptionsByName allKeys] firstObject];

        // Create feature value from pixel buffer
        MLFeatureValue* inputValue = [MLFeatureValue featureValueWithPixelBuffer:pixel_buffer];

        // Create input provider
        MLDictionaryFeatureProvider* inputProvider =
            [[MLDictionaryFeatureProvider alloc]
                initWithDictionary:@{inputName: inputValue}
                             error:&error];

        if (error) {
            throw std::runtime_error("Failed to create input provider: " +
                std::string([[error localizedDescription] UTF8String]));
        }

        // Run prediction
        id<MLFeatureProvider> output = [model predictionFromFeatures:inputProvider
                                                               error:&error];

        if (error) {
            throw std::runtime_error("Inference failed: " +
                std::string([[error localizedDescription] UTF8String]));
        }

        // Get output
        NSString* outputName = [[desc.outputDescriptionsByName allKeys] firstObject];
        MLFeatureValue* outputValue = [output featureValueForName:outputName];
        MLMultiArray* outputArray = outputValue.multiArrayValue;

        return (__bridge_retained void*)outputArray;
    }
}

/**
 * @brief Release MLMultiArray
 */
void release_array(void* array_handle) {
    if (array_handle) {
        MLMultiArray* array = (__bridge_transfer MLMultiArray*)array_handle;
        array = nil;
    }
}

/**
 * @brief Get raw pointer to MLMultiArray data
 */
const float* get_array_data(void* array_handle) {
    if (!array_handle) {
        return nullptr;
    }

    MLMultiArray* array = (__bridge MLMultiArray*)array_handle;
    return static_cast<const float*>(array.dataPointer);
}

/**
 * @brief Get MLMultiArray shape
 */
std::vector<int> get_array_shape(void* array_handle) {
    std::vector<int> shape;

    if (!array_handle) {
        return shape;
    }

    MLMultiArray* array = (__bridge MLMultiArray*)array_handle;
    NSArray<NSNumber*>* nsShape = array.shape;

    for (NSNumber* dim in nsShape) {
        shape.push_back([dim intValue]);
    }

    return shape;
}

} // namespace model_loader
} // namespace yolov12
