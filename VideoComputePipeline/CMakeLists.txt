cmake_minimum_required(VERSION 3.10)
project(VideoComputePipeline VERSION 0.1.0 LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

option(BUILD_TESTING "Build tests" ON)
set(FFMPEG_ROOT "" CACHE PATH "Optional FFmpeg installation root")
set(OPENCL_ROOT "" CACHE PATH "Optional OpenCL installation root")

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
    src/pipeline/pipeline_config.c
    src/pipeline/pipeline_runner.c
    src/pipeline/frame_slot.c
    src/pipeline/frame_queue.c
    src/pipeline/frame_pool.c
    src/benchmark/timer.c
    src/benchmark/benchmark.c
    src/benchmark/matrix_report.c
    src/utils/logger.c
    src/utils/file_utils.c
)

find_package(PkgConfig QUIET)

if(PkgConfig_FOUND)
    pkg_check_modules(FFMPEG QUIET IMPORTED_TARGET libavformat libavcodec libavutil libswscale)
endif()

if(NOT FFMPEG_FOUND)
    if(FFMPEG_ROOT)
        list(APPEND CMAKE_PREFIX_PATH "${FFMPEG_ROOT}")
        include_directories("${FFMPEG_ROOT}/include")
        link_directories("${FFMPEG_ROOT}/lib")
    endif()
    find_library(AVFORMAT_LIBRARY avformat PATHS "${FFMPEG_ROOT}/lib")
    find_library(AVCODEC_LIBRARY avcodec PATHS "${FFMPEG_ROOT}/lib")
    find_library(AVUTIL_LIBRARY avutil PATHS "${FFMPEG_ROOT}/lib")
    find_library(SWSCALE_LIBRARY swscale PATHS "${FFMPEG_ROOT}/lib")
    if(AVFORMAT_LIBRARY AND AVCODEC_LIBRARY AND AVUTIL_LIBRARY AND SWSCALE_LIBRARY)
        set(FFMPEG_LIBRARIES ${AVFORMAT_LIBRARY} ${AVCODEC_LIBRARY} ${AVUTIL_LIBRARY} ${SWSCALE_LIBRARY})
    else()
        set(FFMPEG_LIBRARIES avformat avcodec avutil swscale)
    endif()
endif()

find_package(OpenCL QUIET)
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
    target_link_libraries(vcp_core PUBLIC ${FFMPEG_LIBRARIES})
endif()

if(OpenCL_FOUND)
    target_include_directories(vcp_core PUBLIC ${OpenCL_INCLUDE_DIRS})
    target_link_libraries(vcp_core PUBLIC ${OpenCL_LIBRARIES})
else()
    target_link_libraries(vcp_core PUBLIC ${OpenCL_LIBRARIES})
endif()

if(MSVC)
    target_compile_options(vcp_core PRIVATE /W4)
else()
    target_compile_options(vcp_core PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(VideoComputePipeline main.c)
target_link_libraries(VideoComputePipeline PRIVATE vcp_core)

if(BUILD_TESTING)
    enable_testing()
    file(GLOB_RECURSE TEST_SOURCES CONFIGURE_DEPENDS "tests/**/*_test.c")
    foreach(TEST_FILE IN LISTS TEST_SOURCES)
        get_filename_component(TEST_NAME ${TEST_FILE} NAME_WE)
        add_executable(${TEST_NAME} ${TEST_FILE})
        target_link_libraries(${TEST_NAME} PRIVATE vcp_core)
        add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
    endforeach()
endif()

message(STATUS "Project: ${PROJECT_NAME}")
message(STATUS "C standard: C11")
message(STATUS "FFmpeg found via pkg-config: ${FFMPEG_FOUND}")
message(STATUS "OpenCL found: ${OpenCL_FOUND}")
