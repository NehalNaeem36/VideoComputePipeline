cmake_minimum_required(VERSION 3.10)
project(VideoComputePipeline C)

# Set C standard to C11
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Set output directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

# Add include directory
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Collect all source files
file(GLOB_RECURSE SOURCES
    "src/**/*.c"
)

# Main executable
add_executable(VideoComputePipeline main.c ${SOURCES})

# TODO: Link FFmpeg libraries
# Uncomment and modify paths as needed:
# find_package(FFMPEG REQUIRED)
# target_include_directories(VideoComputePipeline PRIVATE ${FFMPEG_INCLUDE_DIRS})
# target_link_libraries(VideoComputePipeline PRIVATE ${FFMPEG_LIBRARIES})

# TODO: Link OpenCL library
# Uncomment and modify paths as needed:
# find_package(OpenCL REQUIRED)
# target_include_directories(VideoComputePipeline PRIVATE ${OpenCL_INCLUDE_DIRS})
# target_link_libraries(VideoComputePipeline PRIVATE ${OpenCL_LIBRARIES})

# Enable testing
enable_testing()

# Collect and add test executables
file(GLOB_RECURSE TEST_SOURCES "tests/**/*_test.c")

foreach(TEST_FILE ${TEST_SOURCES})
    # Extract test name from file path
    get_filename_component(TEST_NAME ${TEST_FILE} NAME_WE)
    
    # Create executable for test
    add_executable(${TEST_NAME} ${TEST_FILE})
    
    # Add test to test suite
    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
endforeach()

# Print build info
message(STATUS "Project: ${PROJECT_NAME}")
message(STATUS "C Standard: C11")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Source files: ${SOURCES}")
message(STATUS "Test files: ${TEST_SOURCES}")
