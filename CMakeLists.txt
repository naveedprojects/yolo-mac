cmake_minimum_required(VERSION 3.20)
project(yolov12-mac VERSION 1.0.0 LANGUAGES CXX OBJCXX)

# C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_OBJCXX_STANDARD 17)
set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)

# Build options
option(YOLOV12_BUILD_SHARED "Build shared library" ON)
option(YOLOV12_BUILD_STATIC "Build static library" OFF)
option(YOLOV12_BUILD_TOOLS "Build CLI tools" ON)
option(YOLOV12_BUILD_EXAMPLES "Build examples" ON)
option(YOLOV12_BUILD_TESTS "Build tests" OFF)

# Backend options
option(YOLOV12_ENABLE_COREML "Enable CoreML backend" ON)
option(YOLOV12_ENABLE_ONNX "Enable ONNX Runtime backend" ON)
option(YOLOV12_ENABLE_PYTHON "Enable Python backend for advanced quantization" ON)

# Pure C++ mode (disables Python entirely)
option(YOLOV12_PURE_CPP "Build pure C++ version only (no Python)" OFF)

# macOS specific settings
if(APPLE)
    set(CMAKE_OSX_DEPLOYMENT_TARGET "13.0" CACHE STRING "Minimum macOS version")

    # Find required frameworks
    find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
    find_library(COREML_FRAMEWORK CoreML REQUIRED)
    find_library(VISION_FRAMEWORK Vision REQUIRED)
    find_library(METAL_FRAMEWORK Metal REQUIRED)
    find_library(ACCELERATE_FRAMEWORK Accelerate REQUIRED)
    find_library(COREVIDEO_FRAMEWORK CoreVideo REQUIRED)
    find_library(COREGRAPHICS_FRAMEWORK CoreGraphics REQUIRED)

    set(APPLE_FRAMEWORKS
        ${FOUNDATION_FRAMEWORK}
        ${COREML_FRAMEWORK}
        ${VISION_FRAMEWORK}
        ${METAL_FRAMEWORK}
        ${ACCELERATE_FRAMEWORK}
        ${COREVIDEO_FRAMEWORK}
        ${COREGRAPHICS_FRAMEWORK}
    )
else()
    message(FATAL_ERROR "This project only supports macOS")
endif()

# Output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/third_party)

# Third-party dependencies
add_subdirectory(third_party)

# Source files - Core
set(CORE_SOURCES
    src/yolov12.mm
    src/inference/types.cpp
)

# Source files - Inference
set(INFERENCE_SOURCES
    src/inference/detector.mm
    src/inference/detector_coreml.mm
    src/inference/model_loader.mm
    src/inference/preprocessor.mm
    src/inference/postprocessor.cpp
)

# Source files - Conversion
set(CONVERSION_SOURCES
    src/conversion/converter.cpp
    src/conversion/quantization/quantizer.cpp
)

# ONNX Runtime sources (if enabled)
if(YOLOV12_ENABLE_ONNX)
    list(APPEND INFERENCE_SOURCES
        src/inference/detector_onnx.cpp
    )
    add_definitions(-DYOLOV12_ENABLE_ONNX)
endif()

# Python backend sources (if enabled and not pure C++ mode)
if(YOLOV12_ENABLE_PYTHON AND NOT YOLOV12_PURE_CPP)
    list(APPEND CONVERSION_SOURCES
        src/python_bridge/python_interpreter.cpp
    )
    add_definitions(-DYOLOV12_ENABLE_PYTHON)

    # Find Python
    find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
    include_directories(${Python3_INCLUDE_DIRS})
endif()

# Pure C++ mode definition
if(YOLOV12_PURE_CPP)
    add_definitions(-DYOLOV12_PURE_CPP)
    message(STATUS "Building in pure C++ mode (no Python support)")
endif()

# All library sources
set(LIB_SOURCES
    ${CORE_SOURCES}
    ${INFERENCE_SOURCES}
    ${CONVERSION_SOURCES}
)

# Build shared library
if(YOLOV12_BUILD_SHARED)
    add_library(yolov12 SHARED ${LIB_SOURCES})
    target_link_libraries(yolov12
        PUBLIC ${APPLE_FRAMEWORKS}
    )

    if(YOLOV12_ENABLE_ONNX)
        target_link_libraries(yolov12 PUBLIC onnxruntime)
    endif()

    if(YOLOV12_ENABLE_PYTHON AND NOT YOLOV12_PURE_CPP)
        target_link_libraries(yolov12 PRIVATE ${Python3_LIBRARIES})
    endif()

    set_target_properties(yolov12 PROPERTIES
        VERSION ${PROJECT_VERSION}
        SOVERSION 1
        PUBLIC_HEADER "include/yolov12/yolov12.h"
    )
endif()

# Build static library
if(YOLOV12_BUILD_STATIC)
    add_library(yolov12_static STATIC ${LIB_SOURCES})
    target_link_libraries(yolov12_static
        PUBLIC ${APPLE_FRAMEWORKS}
    )

    if(YOLOV12_ENABLE_ONNX)
        target_link_libraries(yolov12_static PUBLIC onnxruntime)
    endif()

    if(YOLOV12_ENABLE_PYTHON AND NOT YOLOV12_PURE_CPP)
        target_link_libraries(yolov12_static PRIVATE ${Python3_LIBRARIES})
    endif()

    set_target_properties(yolov12_static PROPERTIES
        OUTPUT_NAME yolov12
    )
endif()

# Determine which library target to use for tools/examples
if(YOLOV12_BUILD_SHARED)
    set(YOLOV12_LIB yolov12)
else()
    set(YOLOV12_LIB yolov12_static)
endif()

# Build CLI tools
if(YOLOV12_BUILD_TOOLS)
    # Converter CLI
    add_executable(yolov12-convert tools/convert_cli.cpp)
    target_link_libraries(yolov12-convert PRIVATE ${YOLOV12_LIB})

    # Benchmark CLI
    add_executable(yolov12-benchmark tools/benchmark_cli.cpp)
    target_link_libraries(yolov12-benchmark PRIVATE ${YOLOV12_LIB})
endif()

# Build examples
if(YOLOV12_BUILD_EXAMPLES)
    add_executable(basic_detection examples/basic_detection.cpp)
    target_link_libraries(basic_detection PRIVATE ${YOLOV12_LIB})

    add_executable(batch_detection examples/batch_detection.cpp)
    target_link_libraries(batch_detection PRIVATE ${YOLOV12_LIB})
endif()

# Build tests
if(YOLOV12_BUILD_TESTS)
    enable_testing()
    find_package(GTest REQUIRED)

    add_executable(yolov12_tests
        tests/test_detector.cpp
        tests/test_converter.cpp
        tests/test_quantization.cpp
        tests/test_preprocessing.cpp
    )
    target_link_libraries(yolov12_tests
        PRIVATE ${YOLOV12_LIB} GTest::gtest_main
    )

    include(GoogleTest)
    gtest_discover_tests(yolov12_tests)
endif()

# Install targets
install(TARGETS ${YOLOV12_LIB}
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    PUBLIC_HEADER DESTINATION include/yolov12
)

install(DIRECTORY include/yolov12/
    DESTINATION include/yolov12
    FILES_MATCHING PATTERN "*.h"
)

if(YOLOV12_BUILD_TOOLS)
    install(TARGETS yolov12-convert yolov12-benchmark
        RUNTIME DESTINATION bin
    )
endif()

# Print configuration summary
message(STATUS "")
message(STATUS "YOLOv12-Mac Configuration:")
message(STATUS "  Version:        ${PROJECT_VERSION}")
message(STATUS "  Build type:     ${CMAKE_BUILD_TYPE}")
message(STATUS "  macOS target:   ${CMAKE_OSX_DEPLOYMENT_TARGET}")
message(STATUS "")
message(STATUS "  Build shared:   ${YOLOV12_BUILD_SHARED}")
message(STATUS "  Build static:   ${YOLOV12_BUILD_STATIC}")
message(STATUS "  Build tools:    ${YOLOV12_BUILD_TOOLS}")
message(STATUS "  Build examples: ${YOLOV12_BUILD_EXAMPLES}")
message(STATUS "  Build tests:    ${YOLOV12_BUILD_TESTS}")
message(STATUS "")
message(STATUS "  CoreML backend: ${YOLOV12_ENABLE_COREML}")
message(STATUS "  ONNX backend:   ${YOLOV12_ENABLE_ONNX}")
message(STATUS "  Python backend: ${YOLOV12_ENABLE_PYTHON}")
message(STATUS "  Pure C++ mode:  ${YOLOV12_PURE_CPP}")
message(STATUS "")
