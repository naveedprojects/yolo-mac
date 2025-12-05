# Third-party dependencies for YOLOv12-Mac

# ONNX Runtime
if(YOLOV12_ENABLE_ONNX)
    # Option 1: Use system-installed ONNX Runtime
    find_library(ONNXRUNTIME_LIB onnxruntime HINTS /usr/local/lib /opt/homebrew/lib)
    find_path(ONNXRUNTIME_INCLUDE onnxruntime_cxx_api.h HINTS /usr/local/include /opt/homebrew/include)

    if(ONNXRUNTIME_LIB AND ONNXRUNTIME_INCLUDE)
        message(STATUS "Found ONNX Runtime: ${ONNXRUNTIME_LIB}")
        add_library(onnxruntime INTERFACE)
        target_include_directories(onnxruntime INTERFACE ${ONNXRUNTIME_INCLUDE})
        target_link_libraries(onnxruntime INTERFACE ${ONNXRUNTIME_LIB})
    else()
        message(STATUS "ONNX Runtime not found. Please install it:")
        message(STATUS "  brew install onnxruntime")
        message(STATUS "  OR download from https://github.com/microsoft/onnxruntime/releases")
        message(STATUS "")
        message(STATUS "Alternatively, disable ONNX backend:")
        message(STATUS "  cmake .. -DYOLOV12_ENABLE_ONNX=OFF")

        # Create stub target to allow configuration to complete
        add_library(onnxruntime INTERFACE)
        set(YOLOV12_ENABLE_ONNX OFF CACHE BOOL "" FORCE)
    endif()
endif()

# pybind11 (for Python embedding, if needed)
if(YOLOV12_ENABLE_PYTHON AND NOT YOLOV12_PURE_CPP)
    # Try to find pybind11
    find_package(pybind11 QUIET)

    if(NOT pybind11_FOUND)
        message(STATUS "pybind11 not found. Python embedding will use CPython API directly.")
        message(STATUS "For easier Python integration, consider installing pybind11:")
        message(STATUS "  brew install pybind11")
    endif()
endif()

# nlohmann/json for configuration parsing
# We'll use a header-only version
set(JSON_HEADER_URL "https://github.com/nlohmann/json/releases/download/v3.11.3/json.hpp")
set(JSON_HEADER_PATH "${CMAKE_CURRENT_SOURCE_DIR}/nlohmann/json.hpp")

if(NOT EXISTS ${JSON_HEADER_PATH})
    message(STATUS "Downloading nlohmann/json header...")
    file(DOWNLOAD ${JSON_HEADER_URL} ${JSON_HEADER_PATH} STATUS DOWNLOAD_STATUS)
    list(GET DOWNLOAD_STATUS 0 STATUS_CODE)
    if(NOT STATUS_CODE EQUAL 0)
        message(WARNING "Failed to download nlohmann/json. Some features may be limited.")
    endif()
endif()

if(EXISTS ${JSON_HEADER_PATH})
    add_library(nlohmann_json INTERFACE)
    target_include_directories(nlohmann_json INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
else()
    # Create stub
    add_library(nlohmann_json INTERFACE)
endif()

# stb_image for image loading (header-only)
set(STB_IMAGE_URL "https://raw.githubusercontent.com/nothings/stb/master/stb_image.h")
set(STB_IMAGE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/stb/stb_image.h")

if(NOT EXISTS ${STB_IMAGE_PATH})
    file(MAKE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/stb")
    message(STATUS "Downloading stb_image.h...")
    file(DOWNLOAD ${STB_IMAGE_URL} ${STB_IMAGE_PATH} STATUS DOWNLOAD_STATUS)
    list(GET DOWNLOAD_STATUS 0 STATUS_CODE)
    if(NOT STATUS_CODE EQUAL 0)
        message(WARNING "Failed to download stb_image.h")
    endif()
endif()

if(EXISTS ${STB_IMAGE_PATH})
    add_library(stb_image INTERFACE)
    target_include_directories(stb_image INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
else()
    add_library(stb_image INTERFACE)
endif()
