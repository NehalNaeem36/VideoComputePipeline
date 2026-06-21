cmake_minimum_required(VERSION 3.10)
project(VideoComputePipeline VERSION 0.1.0 LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

option(BUILD_TESTING "Build tests" ON)
option(ENABLE_CUDA_INFERENCE "Build CUDA/TensorRT inference backend" OFF)
option(ENABLE_HW_VIDEO "Build NVDEC/NVENC CUDA hardware video path" OFF)
set(FFMPEG_ROOT "" CACHE PATH "Optional FFmpeg installation root")
set(OPENCL_ROOT "" CACHE PATH "Optional OpenCL installation root")
set(TENSORRT_ROOT "" CACHE PATH "Optional TensorRT installation root")
set(VCP_TEST_TENSORRT_ENGINE "" CACHE FILEPATH "Optional TensorRT engine for CUDA inference tests")

function(vcp_reject_msys_path VAR_NAME)
    if(MSVC AND DEFINED ${VAR_NAME} AND ${VAR_NAME})
        string(TOLOWER "${${VAR_NAME}}" VCP_PATH_LOWER)
        if(VCP_PATH_LOWER MATCHES "c:[/\\\\]msys64|[/\\\\]msys64[/\\\\]|[/\\\\](ucrt64|mingw64|mingw32)[/\\\\]")
            message(FATAL_ERROR
                "${VAR_NAME} points to a MinGW/MSYS2 path: ${${VAR_NAME}}\n"
                "MSVC builds must not use MSYS2 headers or .a libraries. "
                "Delete build-msvc and configure with MSVC-compatible dependencies.")
        endif()
    endif()
endfunction()

if(ENABLE_HW_VIDEO AND NOT ENABLE_CUDA_INFERENCE)
    message(FATAL_ERROR "ENABLE_HW_VIDEO=ON requires ENABLE_CUDA_INFERENCE=ON")
endif()

if(ENABLE_CUDA_INFERENCE)
    enable_language(CXX CUDA)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    find_package(CUDAToolkit REQUIRED)

    find_path(TENSORRT_INCLUDE_DIR NvInfer.h
        HINTS
            "${TENSORRT_ROOT}/include"
            "$ENV{TENSORRT_ROOT}/include"
            "$ENV{TensorRT_ROOT}/include"
    )
    find_library(TENSORRT_NVINFER_LIBRARY
        NAMES nvinfer nvinfer_11
        HINTS
            "${TENSORRT_ROOT}/lib"
            "${TENSORRT_ROOT}/lib/x64"
            "$ENV{TENSORRT_ROOT}/lib"
            "$ENV{TENSORRT_ROOT}/lib/x64"
            "$ENV{TensorRT_ROOT}/lib"
            "$ENV{TensorRT_ROOT}/lib/x64"
    )
    if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_NVINFER_LIBRARY)
        message(FATAL_ERROR "ENABLE_CUDA_INFERENCE=ON requires TensorRT. Set -DTENSORRT_ROOT=C:/path/to/TensorRT from a VS Developer Prompt.")
    endif()
endif()

set(PROJECT_INCLUDE_DIRS
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

set(PROJECT_SOURCES
    src/core/frame.c
    src/video/video_reader.c
    src/video/video_writer.c
    src/cpu/cpu_filters.c
    src/gpu/opencl_context.c
    src/gpu/opencl_program.c
    src/gpu/gpu_filters.c
    src/gpu/cuda_frame.c
    src/pipeline/pipeline_config.c
    src/pipeline/pipeline_runner.c
    src/pipeline/frame_slot.c
    src/pipeline/frame_queue.c
    src/pipeline/frame_pool.c
    src/benchmark/timer.c
    src/benchmark/benchmark.c
    src/benchmark/matrix_report.c
    src/inference/detection_result.c
    src/inference/detection_writer.c
    src/utils/logger.c
    src/utils/file_utils.c
)

if(ENABLE_CUDA_INFERENCE)
    list(APPEND PROJECT_SOURCES
        src/inference/inference_engine_tensorrt.cpp
        src/inference/yolo_postprocess.cpp
        src/inference/cuda_preprocess.cu
    )
else()
    list(APPEND PROJECT_SOURCES
        src/inference/inference_engine_stub.c
    )
endif()

if(ENABLE_HW_VIDEO)
    list(APPEND PROJECT_SOURCES
        src/gpu/cuda_overlay.cu
        src/video/video_hw_reader.cpp
        src/video/video_hw_writer.cpp
    )
else()
    list(APPEND PROJECT_SOURCES
        src/gpu/cuda_overlay_stub.c
        src/video/video_hw_reader_stub.c
        src/video/video_hw_writer_stub.c
    )
endif()

if(NOT MSVC)
    find_package(PkgConfig QUIET)
endif()

if(PkgConfig_FOUND AND NOT MSVC)
    pkg_check_modules(FFMPEG QUIET IMPORTED_TARGET libavformat libavcodec libavutil libswscale)
endif()

if(NOT FFMPEG_FOUND)
    if(MSVC)
        find_path(FFMPEG_INCLUDE_DIR libavformat/avformat.h
            HINTS
                "${FFMPEG_ROOT}/include"
                "$ENV{FFMPEG_ROOT}/include"
            NO_DEFAULT_PATH
        )
        find_library(AVFORMAT_LIBRARY avformat PATHS "${FFMPEG_ROOT}/lib" "$ENV{FFMPEG_ROOT}/lib" NO_DEFAULT_PATH)
        find_library(AVCODEC_LIBRARY avcodec PATHS "${FFMPEG_ROOT}/lib" "$ENV{FFMPEG_ROOT}/lib" NO_DEFAULT_PATH)
        find_library(AVUTIL_LIBRARY avutil PATHS "${FFMPEG_ROOT}/lib" "$ENV{FFMPEG_ROOT}/lib" NO_DEFAULT_PATH)
        find_library(SWSCALE_LIBRARY swscale PATHS "${FFMPEG_ROOT}/lib" "$ENV{FFMPEG_ROOT}/lib" NO_DEFAULT_PATH)
    else()
        find_path(FFMPEG_INCLUDE_DIR libavformat/avformat.h
            HINTS
                "${FFMPEG_ROOT}/include"
                "$ENV{FFMPEG_ROOT}/include"
        )
        find_library(AVFORMAT_LIBRARY avformat PATHS "${FFMPEG_ROOT}/lib" "$ENV{FFMPEG_ROOT}/lib")
        find_library(AVCODEC_LIBRARY avcodec PATHS "${FFMPEG_ROOT}/lib" "$ENV{FFMPEG_ROOT}/lib")
        find_library(AVUTIL_LIBRARY avutil PATHS "${FFMPEG_ROOT}/lib" "$ENV{FFMPEG_ROOT}/lib")
        find_library(SWSCALE_LIBRARY swscale PATHS "${FFMPEG_ROOT}/lib" "$ENV{FFMPEG_ROOT}/lib")
    endif()
    vcp_reject_msys_path(FFMPEG_INCLUDE_DIR)
    vcp_reject_msys_path(AVFORMAT_LIBRARY)
    vcp_reject_msys_path(AVCODEC_LIBRARY)
    vcp_reject_msys_path(AVUTIL_LIBRARY)
    vcp_reject_msys_path(SWSCALE_LIBRARY)
    if(MSVC AND (NOT FFMPEG_INCLUDE_DIR OR NOT AVFORMAT_LIBRARY OR NOT AVCODEC_LIBRARY OR NOT AVUTIL_LIBRARY OR NOT SWSCALE_LIBRARY))
        message(FATAL_ERROR "MSVC builds cannot use MSYS2 FFmpeg headers/libs. Install or provide an MSVC-compatible FFmpeg development package and configure with -DFFMPEG_ROOT=C:/path/to/ffmpeg-dev. Do not point FFMPEG_ROOT at C:/msys64/ucrt64.")
    endif()
    if(FFMPEG_INCLUDE_DIR)
        set(FFMPEG_INCLUDE_DIRS "${FFMPEG_INCLUDE_DIR}")
    endif()
    if(AVFORMAT_LIBRARY AND AVCODEC_LIBRARY AND AVUTIL_LIBRARY AND SWSCALE_LIBRARY)
        set(FFMPEG_LIBRARIES ${AVFORMAT_LIBRARY} ${AVCODEC_LIBRARY} ${AVUTIL_LIBRARY} ${SWSCALE_LIBRARY})
    else()
        set(FFMPEG_LIBRARIES avformat avcodec avutil swscale)
    endif()
endif()

if(MSVC)
    find_path(OpenCL_INCLUDE_DIRS CL/cl.h
        HINTS
            "${OPENCL_ROOT}/include"
            "$ENV{OPENCL_ROOT}/include"
            "${CUDAToolkit_TARGET_DIR}/include"
            "${CUDAToolkit_INCLUDE_DIRS}"
        NO_DEFAULT_PATH
    )
    find_library(OpenCL_LIBRARIES OpenCL
        HINTS
            "${OPENCL_ROOT}/lib/x64"
            "${OPENCL_ROOT}/lib"
            "$ENV{OPENCL_ROOT}/lib/x64"
            "$ENV{OPENCL_ROOT}/lib"
            "${CUDAToolkit_TARGET_DIR}/lib/x64"
            "${CUDAToolkit_LIBRARY_DIR}"
        NO_DEFAULT_PATH
    )
    if(NOT OpenCL_INCLUDE_DIRS OR NOT OpenCL_LIBRARIES)
        message(FATAL_ERROR "MSVC builds require MSVC-compatible OpenCL headers/libs. Set -DOPENCL_ROOT=C:/path/to/OpenCL or build with CUDA Toolkit available.")
    endif()
    set(OpenCL_FOUND TRUE)
else()
    find_package(OpenCL QUIET)
endif()
if(NOT OpenCL_FOUND)
    if(OPENCL_ROOT)
        include_directories("${OPENCL_ROOT}/include")
        link_directories("${OPENCL_ROOT}/lib")
    endif()
    find_library(OpenCL_LIBRARY OpenCL PATHS "${OPENCL_ROOT}/lib")
    if(OpenCL_LIBRARY)
        set(OpenCL_LIBRARIES ${OpenCL_LIBRARY})
    else()
        set(OpenCL_LIBRARIES OpenCL)
    endif()
endif()

add_library(vcp_core STATIC ${PROJECT_SOURCES})
target_include_directories(vcp_core PUBLIC ${PROJECT_INCLUDE_DIRS})
target_compile_definitions(vcp_core PUBLIC
    VCP_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
    VCP_KERNEL_DIR="${CMAKE_CURRENT_SOURCE_DIR}/kernels"
)

if(FFMPEG_FOUND AND TARGET PkgConfig::FFMPEG)
    target_link_libraries(vcp_core PUBLIC PkgConfig::FFMPEG)
else()
    if(FFMPEG_INCLUDE_DIRS)
        target_include_directories(vcp_core PUBLIC ${FFMPEG_INCLUDE_DIRS})
    endif()
    target_link_libraries(vcp_core PUBLIC ${FFMPEG_LIBRARIES})
endif()

if(OpenCL_FOUND)
    target_include_directories(vcp_core PUBLIC ${OpenCL_INCLUDE_DIRS})
    target_link_libraries(vcp_core PUBLIC ${OpenCL_LIBRARIES})
else()
    target_link_libraries(vcp_core PUBLIC ${OpenCL_LIBRARIES})
endif()

if(ENABLE_CUDA_INFERENCE)
    target_compile_definitions(vcp_core PUBLIC VCP_ENABLE_CUDA_INFERENCE=1)
    target_include_directories(vcp_core PUBLIC ${TENSORRT_INCLUDE_DIR})
    target_link_libraries(vcp_core PUBLIC CUDA::cudart ${TENSORRT_NVINFER_LIBRARY})
endif()

if(ENABLE_HW_VIDEO)
    target_compile_definitions(vcp_core PUBLIC VCP_ENABLE_HW_VIDEO=1)
endif()

if(MSVC)
    target_compile_options(vcp_core PRIVATE
        $<$<COMPILE_LANGUAGE:C>:/W4>
        $<$<COMPILE_LANGUAGE:CXX>:/W4>
    )
else()
    target_compile_options(vcp_core PRIVATE
        $<$<COMPILE_LANGUAGE:C>:-Wall -Wextra -Wpedantic>
        $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>
    )
endif()

add_executable(VideoComputePipeline main.c)
target_link_libraries(VideoComputePipeline PRIVATE vcp_core)
if(ENABLE_CUDA_INFERENCE)
    set_target_properties(VideoComputePipeline PROPERTIES LINKER_LANGUAGE CXX)
endif()

if(BUILD_TESTING)
    enable_testing()
    file(GLOB_RECURSE TEST_SOURCES CONFIGURE_DEPENDS "tests/**/*_test.c")
    foreach(TEST_FILE IN LISTS TEST_SOURCES)
        get_filename_component(TEST_NAME ${TEST_FILE} NAME_WE)
        add_executable(${TEST_NAME} ${TEST_FILE})
        target_link_libraries(${TEST_NAME} PRIVATE vcp_core)
        if(ENABLE_CUDA_INFERENCE)
            set_target_properties(${TEST_NAME} PROPERTIES LINKER_LANGUAGE CXX)
            target_compile_definitions(${TEST_NAME} PRIVATE VCP_TEST_TENSORRT_ENGINE="${VCP_TEST_TENSORRT_ENGINE}")
        endif()
        add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
    endforeach()
endif()

message(STATUS "Project: ${PROJECT_NAME}")
message(STATUS "C standard: C11")
message(STATUS "FFmpeg found via pkg-config: ${FFMPEG_FOUND}")
message(STATUS "OpenCL found: ${OpenCL_FOUND}")
message(STATUS "CUDA inference enabled: ${ENABLE_CUDA_INFERENCE}")
message(STATUS "Hardware video enabled: ${ENABLE_HW_VIDEO}")
if(ENABLE_CUDA_INFERENCE)
    message(STATUS "TensorRT include: ${TENSORRT_INCLUDE_DIR}")
    message(STATUS "TensorRT nvinfer: ${TENSORRT_NVINFER_LIBRARY}")
endif()
