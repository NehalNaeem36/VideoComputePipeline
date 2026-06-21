cmake_minimum_required(VERSION 3.20)

project(VideoComputePipelineUI LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(SDL2 CONFIG REQUIRED)
find_package(OpenGL REQUIRED)

set(IMGUI_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/imgui")

add_executable(VideoComputePipelineUI
    src/main.cpp
    src/ui_app.cpp
    src/command_builder.cpp
    src/process_runner_win32.cpp
    src/progress_parser.cpp
    src/path_utils.cpp
    ${IMGUI_DIR}/imgui.cpp
    ${IMGUI_DIR}/imgui_draw.cpp
    ${IMGUI_DIR}/imgui_tables.cpp
    ${IMGUI_DIR}/imgui_widgets.cpp
    ${IMGUI_DIR}/backends/imgui_impl_sdl2.cpp
    ${IMGUI_DIR}/backends/imgui_impl_opengl3.cpp
)

target_include_directories(VideoComputePipelineUI PRIVATE
    include
    ${IMGUI_DIR}
    ${IMGUI_DIR}/backends
)

target_compile_definitions(VideoComputePipelineUI PRIVATE SDL_MAIN_HANDLED NOMINMAX WIN32_LEAN_AND_MEAN)

if(TARGET SDL2::SDL2)
    target_link_libraries(VideoComputePipelineUI PRIVATE SDL2::SDL2)
elseif(TARGET SDL2::SDL2-static)
    target_link_libraries(VideoComputePipelineUI PRIVATE SDL2::SDL2-static)
else()
    message(FATAL_ERROR "SDL2 CMake target was not found. Install SDL2 with vcpkg.")
endif()

target_link_libraries(VideoComputePipelineUI PRIVATE OpenGL::GL)

if(MSVC)
    target_compile_options(VideoComputePipelineUI PRIVATE /W4 /permissive-)
else()
    target_compile_options(VideoComputePipelineUI PRIVATE -Wall -Wextra -Wpedantic)
endif()

set_target_properties(VideoComputePipelineUI PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/Release"
    RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/Debug"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/Release"
    RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO "${CMAKE_BINARY_DIR}/Release"
    RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL "${CMAKE_BINARY_DIR}/Release"
)
