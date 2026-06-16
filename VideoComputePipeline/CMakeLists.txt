cmake_minimum_required(VERSION 3.10)
project(VideoComputePipeline VERSION 0.1.0 LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

set(PROJECT_INCLUDE_DIRS
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

set(MILESTONE_SOURCES
    src/core/frame.c
    src/benchmark/timer.c
    src/utils/logger.c
    src/pipeline/pipeline_config.c
    src/pipeline/pipeline_runner.c
)

add_library(vcp_core STATIC ${MILESTONE_SOURCES})
target_include_directories(vcp_core PUBLIC ${PROJECT_INCLUDE_DIRS})

if(MSVC)
    target_compile_options(vcp_core PRIVATE /W4)
else()
    target_compile_options(vcp_core PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(VideoComputePipeline main.c)
target_link_libraries(VideoComputePipeline PRIVATE vcp_core)

if(MSVC)
    target_compile_options(VideoComputePipeline PRIVATE /W4)
else()
    target_compile_options(VideoComputePipeline PRIVATE -Wall -Wextra -Wpedantic)
endif()

enable_testing()

set(MILESTONE_TESTS
    tests/core/frame_test.c
    tests/benchmark/timer_test.c
    tests/utils/logger_test.c
    tests/pipeline/pipeline_config_test.c
    tests/pipeline/pipeline_runner_test.c
)

foreach(TEST_FILE IN LISTS MILESTONE_TESTS)
    get_filename_component(TEST_NAME ${TEST_FILE} NAME_WE)
    add_executable(${TEST_NAME} ${TEST_FILE})
    target_link_libraries(${TEST_NAME} PRIVATE vcp_core)
    if(MSVC)
        target_compile_options(${TEST_NAME} PRIVATE /W4)
    else()
        target_compile_options(${TEST_NAME} PRIVATE -Wall -Wextra -Wpedantic)
    endif()
    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
endforeach()

message(STATUS "Project: ${PROJECT_NAME}")
message(STATUS "C Standard: C11")
message(STATUS "Milestone sources: ${MILESTONE_SOURCES}")
message(STATUS "Milestone tests: ${MILESTONE_TESTS}")
