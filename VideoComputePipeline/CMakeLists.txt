cmake_minimum_required(VERSION 3.10)
project(VideoComputePipeline VERSION 0.1.0 LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

if(CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/Release")
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/Release")
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/lib/Release")
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/Debug")
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/Debug")
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/lib/Debug")
endif()

option(BUILD_TESTING "Build tests" ON)
option(ENABLE_TENSORRT "Build TensorRT inference runtime when CUDA inference is enabled" ON)
option(ENABLE_CUDA_INFERENCE "Build CUDA/TensorRT inference backend" OFF)
option(ENABLE_HW_VIDEO "Build NVDEC/NVENC CUDA hardware video path" OFF)
option(ENABLE_ONNXRUNTIME "Build ONNX Runtime inference backend" OFF)
option(ENABLE_LIBTORCH "Build LibTorch TorchScript inference backend" OFF)
set(FFMPEG_ROOT "" CACHE PATH "Optional FFmpeg installation root")
set(OPENCL_ROOT "" CACHE PATH "Optional OpenCL installation root")
set(TENSORRT_ROOT "" CACHE PATH "Optional TensorRT installation root")
set(CUDNN_ROOT "" CACHE PATH "Optional cuDNN installation root")
set(ONNXRUNTIME_ROOT "" CACHE PATH "Optional ONNX Runtime installation root")
set(LIBTORCH_ROOT "" CACHE PATH "Optional LibTorch installation root")
set(VCP_TEST_TENSORRT_ENGINE "" CACHE FILEPATH "Optional TensorRT engine for CUDA inference tests")

function(vcp_reject_msys_path VAR_NAME)
    if(MSVC AND DEFINED ${VAR_NAME} AND ${VAR_NAME})
        string(TOLOWER "${${VAR_NAME}}" VCP_PATH_LOWER)
        if(VCP_PATH_LOWER MATCHES "c:[/\\\\]msys64|[/\\\\]msys64[/\\\\]|[/\\\\](ucrt64|mingw64|mingw32)[/\\\\]")
            message(FATAL_ERROR
                "${VAR_NAME} points to a MinGW/MSYS2 path: ${${VAR_NAME}}\n"
                "MSVC builds must not use MSYS2 headers or .a libraries. "
                "Delete the MSVC build directory and configure with MSVC-compatible dependencies.")
        endif()
    endif()
endfunction()

function(vcp_reject_cuda13_path VAR_NAME)
    if(DEFINED ${VAR_NAME} AND ${VAR_NAME})
        string(TOLOWER "${${VAR_NAME}}" VCP_PATH_LOWER)
        if(VCP_PATH_LOWER MATCHES "cuda[/\\\\]v13|cuda[/\\\\]v13\\.|cuda-13|cuda13|13\\.3")
            message(FATAL_ERROR
                "${VAR_NAME} appears to point at the old CUDA 13 stack: ${${VAR_NAME}}\n"
                "Configure the CUDA inference build with the CUDA 12.9 stack paths from prompt-file.txt.")
        endif()
    endif()
endfunction()

function(vcp_require_file VAR_NAME DESCRIPTION)
    if(NOT DEFINED ${VAR_NAME} OR NOT ${VAR_NAME} OR NOT EXISTS "${${VAR_NAME}}")
        message(FATAL_ERROR "Missing required ${DESCRIPTION}: ${${VAR_NAME}}")
    endif()
endfunction()

function(vcp_copy_runtime_dlls TARGET_NAME)
    foreach(VCP_DLL IN LISTS ARGN)
        if(VCP_DLL AND EXISTS "${VCP_DLL}")
            add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${VCP_DLL}"
                    "$<TARGET_FILE_DIR:${TARGET_NAME}>"
                VERBATIM)
        endif()
    endforeach()
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
    if(NOT CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 86 CACHE STRING "CUDA architectures for VideoComputePipeline" FORCE)
    endif()
    vcp_reject_cuda13_path(CUDAToolkit_ROOT)
    vcp_reject_cuda13_path(CUDA_TOOLKIT_ROOT_DIR)
    vcp_reject_cuda13_path(OPENCL_ROOT)
    vcp_reject_cuda13_path(TENSORRT_ROOT)
    find_package(CUDAToolkit REQUIRED)

    if(ENABLE_TENSORRT)
        find_path(TENSORRT_INCLUDE_DIR NvInfer.h
            HINTS
                "${TENSORRT_ROOT}/include"
                "$ENV{TENSORRT_ROOT}/include"
                "$ENV{TensorRT_ROOT}/include"
            NO_DEFAULT_PATH
        )
        find_path(TENSORRT_ONNX_PARSER_INCLUDE_DIR NvOnnxParser.h
            HINTS
                "${TENSORRT_ROOT}/include"
                "$ENV{TENSORRT_ROOT}/include"
                "$ENV{TensorRT_ROOT}/include"
            NO_DEFAULT_PATH
        )
        find_library(TENSORRT_NVINFER_LIBRARY
            NAMES nvinfer_11 nvinfer
            HINTS
                "${TENSORRT_ROOT}/lib"
                "${TENSORRT_ROOT}/lib/x64"
                "$ENV{TENSORRT_ROOT}/lib"
                "$ENV{TENSORRT_ROOT}/lib/x64"
                "$ENV{TensorRT_ROOT}/lib"
                "$ENV{TensorRT_ROOT}/lib/x64"
            NO_DEFAULT_PATH
        )
        find_library(TENSORRT_NVINFER_PLUGIN_LIBRARY
            NAMES nvinfer_plugin_11 nvinfer_plugin
            HINTS
                "${TENSORRT_ROOT}/lib"
                "${TENSORRT_ROOT}/lib/x64"
                "$ENV{TENSORRT_ROOT}/lib"
                "$ENV{TENSORRT_ROOT}/lib/x64"
                "$ENV{TensorRT_ROOT}/lib"
                "$ENV{TensorRT_ROOT}/lib/x64"
            NO_DEFAULT_PATH
        )
        find_library(TENSORRT_NVONNXPARSER_LIBRARY
            NAMES nvonnxparser_11 nvonnxparser
            HINTS
                "${TENSORRT_ROOT}/lib"
                "${TENSORRT_ROOT}/lib/x64"
                "$ENV{TENSORRT_ROOT}/lib"
                "$ENV{TENSORRT_ROOT}/lib/x64"
                "$ENV{TensorRT_ROOT}/lib"
                "$ENV{TensorRT_ROOT}/lib/x64"
            NO_DEFAULT_PATH
        )
        find_file(TENSORRT_NVINFER_DLL
            NAMES nvinfer_11.dll
            HINTS "${TENSORRT_ROOT}/bin" "$ENV{TENSORRT_ROOT}/bin" "$ENV{TensorRT_ROOT}/bin"
            NO_DEFAULT_PATH
        )
        find_file(TENSORRT_NVINFER_PLUGIN_DLL
            NAMES nvinfer_plugin_11.dll
            HINTS "${TENSORRT_ROOT}/bin" "$ENV{TENSORRT_ROOT}/bin" "$ENV{TensorRT_ROOT}/bin"
            NO_DEFAULT_PATH
        )
        find_file(TENSORRT_NVONNXPARSER_DLL
            NAMES nvonnxparser_11.dll
            HINTS "${TENSORRT_ROOT}/bin" "$ENV{TENSORRT_ROOT}/bin" "$ENV{TensorRT_ROOT}/bin"
            NO_DEFAULT_PATH
        )
        if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_ONNX_PARSER_INCLUDE_DIR OR
           NOT TENSORRT_NVINFER_LIBRARY OR NOT TENSORRT_NVINFER_PLUGIN_LIBRARY OR NOT TENSORRT_NVONNXPARSER_LIBRARY OR
           NOT TENSORRT_NVINFER_DLL OR NOT TENSORRT_NVINFER_PLUGIN_DLL OR NOT TENSORRT_NVONNXPARSER_DLL)
            message(FATAL_ERROR
                "ENABLE_TENSORRT=ON requires TensorRT 11.1 CUDA 12 files. Set -DTENSORRT_ROOT=D:/TensorRT/TensorRT-11.1.0.106.\n"
                "Required: include/NvInfer.h, include/NvOnnxParser.h, lib/nvinfer_11.lib, lib/nvinfer_plugin_11.lib, lib/nvonnxparser_11.lib, and matching bin/*.dll files.")
        endif()
    endif()

    find_path(CUDNN_INCLUDE_DIR cudnn.h
        HINTS "${CUDNN_ROOT}/include" "$ENV{CUDNN_ROOT}/include"
        NO_DEFAULT_PATH
    )
    find_library(CUDNN_LIBRARY
        NAMES cudnn
        HINTS "${CUDNN_ROOT}/lib/x64" "$ENV{CUDNN_ROOT}/lib/x64"
        NO_DEFAULT_PATH
    )
    find_file(CUDNN_RUNTIME_DLL
        NAMES cudnn64_9.dll
        HINTS "${CUDNN_ROOT}/bin/x64" "$ENV{CUDNN_ROOT}/bin/x64"
        NO_DEFAULT_PATH
    )
    if(NOT CUDNN_INCLUDE_DIR OR NOT CUDNN_LIBRARY OR NOT CUDNN_RUNTIME_DLL)
        message(FATAL_ERROR
            "ENABLE_CUDA_INFERENCE=ON requires cuDNN 9 for CUDA 12. Set -DCUDNN_ROOT=D:/cuDNN/cudnn-windows-x86_64-9.23.2.1_cuda12-archive.\n"
            "Required layout: include, lib/x64, and bin/x64/cudnn64_9.dll.")
    endif()

    if(ENABLE_ONNXRUNTIME)
        find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
            HINTS
                "${ONNXRUNTIME_ROOT}/buildTransitive/native/include"
                "$ENV{ONNXRUNTIME_ROOT}/include"
                "$ENV{ONNXRUNTIME_ROOT}/buildTransitive/native/include"
                "$ENV{OnnxRuntime_ROOT}/include"
                "$ENV{OnnxRuntime_ROOT}/buildTransitive/native/include"
            NO_DEFAULT_PATH
        )
        find_library(ONNXRUNTIME_LIBRARY
            NAMES onnxruntime
            HINTS
                "${ONNXRUNTIME_ROOT}/runtimes/win-x64/native"
                "$ENV{ONNXRUNTIME_ROOT}/runtimes/win-x64/native"
                "$ENV{OnnxRuntime_ROOT}/runtimes/win-x64/native"
            NO_DEFAULT_PATH
        )
        find_file(ONNXRUNTIME_DLL
            NAMES onnxruntime.dll
            HINTS "${ONNXRUNTIME_ROOT}/runtimes/win-x64/native" "$ENV{ONNXRUNTIME_ROOT}/runtimes/win-x64/native" "$ENV{OnnxRuntime_ROOT}/runtimes/win-x64/native"
            NO_DEFAULT_PATH
        )
        find_file(ONNXRUNTIME_PROVIDERS_CUDA_DLL
            NAMES onnxruntime_providers_cuda.dll
            HINTS "${ONNXRUNTIME_ROOT}/runtimes/win-x64/native" "$ENV{ONNXRUNTIME_ROOT}/runtimes/win-x64/native" "$ENV{OnnxRuntime_ROOT}/runtimes/win-x64/native"
            NO_DEFAULT_PATH
        )
        find_file(ONNXRUNTIME_PROVIDERS_SHARED_DLL
            NAMES onnxruntime_providers_shared.dll
            HINTS "${ONNXRUNTIME_ROOT}/runtimes/win-x64/native" "$ENV{ONNXRUNTIME_ROOT}/runtimes/win-x64/native" "$ENV{OnnxRuntime_ROOT}/runtimes/win-x64/native"
            NO_DEFAULT_PATH
        )
        if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIBRARY OR
           NOT ONNXRUNTIME_DLL OR NOT ONNXRUNTIME_PROVIDERS_CUDA_DLL OR NOT ONNXRUNTIME_PROVIDERS_SHARED_DLL)
            message(FATAL_ERROR
                "ENABLE_ONNXRUNTIME=ON requires the ONNX Runtime GPU 1.26 Windows NuGet layout.\n"
                "Set -DONNXRUNTIME_ROOT=E:/wAI/third_party/onnxruntime/Microsoft.ML.OnnxRuntime.Gpu.Windows.1.26.0.\n"
                "Required: buildTransitive/native/include/onnxruntime_cxx_api.h and runtimes/win-x64/native/onnxruntime*.dll/lib.")
        endif()
    endif()

    if(ENABLE_LIBTORCH)
        if(LIBTORCH_ROOT)
            list(APPEND CMAKE_PREFIX_PATH "${LIBTORCH_ROOT}")
        endif()
        find_package(Torch REQUIRED)
        if(NOT Torch_FOUND)
            message(FATAL_ERROR "ENABLE_LIBTORCH=ON requires LibTorch. Set -DLIBTORCH_ROOT=C:/path/to/libtorch or add it to CMAKE_PREFIX_PATH.")
        endif()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
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
    src/pipeline/frame_batch.c
    src/pipeline/video_profile.c
    src/pipeline/hardware_profile.c
    src/pipeline/pipeline_execution_plan.c
    src/benchmark/timer.c
    src/benchmark/benchmark.c
    src/benchmark/matrix_report.c
    src/inference/backend_registry.c
    src/inference/detection_result.c
    src/inference/detection_writer.c
    src/utils/logger.c
    src/utils/file_utils.c
)

if(ENABLE_CUDA_INFERENCE AND (ENABLE_TENSORRT OR ENABLE_ONNXRUNTIME OR ENABLE_LIBTORCH))
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
                "C:/vcpkg/installed/x64-windows/include"
            NO_DEFAULT_PATH
        )
        find_library(AVFORMAT_LIBRARY avformat PATHS "${FFMPEG_ROOT}/lib" "$ENV{FFMPEG_ROOT}/lib" "C:/vcpkg/installed/x64-windows/lib" NO_DEFAULT_PATH)
        find_library(AVCODEC_LIBRARY avcodec PATHS "${FFMPEG_ROOT}/lib" "$ENV{FFMPEG_ROOT}/lib" "C:/vcpkg/installed/x64-windows/lib" NO_DEFAULT_PATH)
        find_library(AVUTIL_LIBRARY avutil PATHS "${FFMPEG_ROOT}/lib" "$ENV{FFMPEG_ROOT}/lib" "C:/vcpkg/installed/x64-windows/lib" NO_DEFAULT_PATH)
        find_library(SWSCALE_LIBRARY swscale PATHS "${FFMPEG_ROOT}/lib" "$ENV{FFMPEG_ROOT}/lib" "C:/vcpkg/installed/x64-windows/lib" NO_DEFAULT_PATH)
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
endif()

if(ENABLE_CUDA_INFERENCE AND ENABLE_TENSORRT)
    target_compile_definitions(vcp_core PUBLIC VCP_ENABLE_TENSORRT=1)
    target_include_directories(vcp_core PUBLIC ${TENSORRT_INCLUDE_DIR})
    target_link_libraries(vcp_core PUBLIC
        CUDA::cudart
        ${TENSORRT_NVINFER_LIBRARY}
        ${TENSORRT_NVINFER_PLUGIN_LIBRARY}
        ${TENSORRT_NVONNXPARSER_LIBRARY}
    )
endif()

if(ENABLE_CUDA_INFERENCE AND ENABLE_ONNXRUNTIME)
    target_compile_definitions(vcp_core PUBLIC VCP_ENABLE_ONNXRUNTIME=1)
    target_include_directories(vcp_core PUBLIC ${ONNXRUNTIME_INCLUDE_DIR})
    target_link_libraries(vcp_core PUBLIC ${ONNXRUNTIME_LIBRARY})
endif()

if(ENABLE_CUDA_INFERENCE AND ENABLE_LIBTORCH)
    target_compile_definitions(vcp_core PUBLIC VCP_ENABLE_LIBTORCH=1)
    target_include_directories(vcp_core PUBLIC ${TORCH_INCLUDE_DIRS})
    target_link_libraries(vcp_core PUBLIC ${TORCH_LIBRARIES})
    if(MSVC)
        target_compile_options(vcp_core PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/bigobj>)
    endif()
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
if(ENABLE_CUDA_INFERENCE AND ENABLE_TENSORRT)
    vcp_copy_runtime_dlls(VideoComputePipeline
        "${TENSORRT_NVINFER_DLL}"
        "${TENSORRT_NVINFER_PLUGIN_DLL}"
        "${TENSORRT_NVONNXPARSER_DLL}"
    )
endif()
if(ENABLE_CUDA_INFERENCE AND ENABLE_ONNXRUNTIME)
    vcp_copy_runtime_dlls(VideoComputePipeline
        "${ONNXRUNTIME_DLL}"
        "${ONNXRUNTIME_PROVIDERS_CUDA_DLL}"
        "${ONNXRUNTIME_PROVIDERS_SHARED_DLL}"
    )
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
        if(ENABLE_CUDA_INFERENCE AND ENABLE_TENSORRT)
            vcp_copy_runtime_dlls(${TEST_NAME}
                "${TENSORRT_NVINFER_DLL}"
                "${TENSORRT_NVINFER_PLUGIN_DLL}"
                "${TENSORRT_NVONNXPARSER_DLL}"
            )
        endif()
        if(ENABLE_CUDA_INFERENCE AND ENABLE_ONNXRUNTIME)
            vcp_copy_runtime_dlls(${TEST_NAME}
                "${ONNXRUNTIME_DLL}"
                "${ONNXRUNTIME_PROVIDERS_CUDA_DLL}"
                "${ONNXRUNTIME_PROVIDERS_SHARED_DLL}"
            )
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
    message(STATUS "CUDA Toolkit: ${CUDAToolkit_VERSION} at ${CUDAToolkit_TARGET_DIR}")
    message(STATUS "cuDNN include: ${CUDNN_INCLUDE_DIR}")
    message(STATUS "cuDNN library: ${CUDNN_LIBRARY}")
    message(STATUS "cuDNN runtime: ${CUDNN_RUNTIME_DLL}")
    if(ENABLE_TENSORRT)
        message(STATUS "TensorRT include: ${TENSORRT_INCLUDE_DIR}")
        message(STATUS "TensorRT nvinfer: ${TENSORRT_NVINFER_LIBRARY}")
        message(STATUS "TensorRT plugin: ${TENSORRT_NVINFER_PLUGIN_LIBRARY}")
        message(STATUS "TensorRT onnx parser: ${TENSORRT_NVONNXPARSER_LIBRARY}")
    endif()
    if(ENABLE_ONNXRUNTIME)
        message(STATUS "ONNX Runtime include: ${ONNXRUNTIME_INCLUDE_DIR}")
        message(STATUS "ONNX Runtime library: ${ONNXRUNTIME_LIBRARY}")
        message(STATUS "ONNX Runtime DLL: ${ONNXRUNTIME_DLL}")
        message(STATUS "ONNX Runtime CUDA provider: ${ONNXRUNTIME_PROVIDERS_CUDA_DLL}")
    endif()
    if(ENABLE_LIBTORCH)
        message(STATUS "LibTorch libraries: ${TORCH_LIBRARIES}")
    endif()
endif()
