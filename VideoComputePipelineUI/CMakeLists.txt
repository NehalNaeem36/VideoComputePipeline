cmake_minimum_required(VERSION 3.20)

project(VideoComputePipelineUI LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(WIN32)
    find_package(SDL2 CONFIG REQUIRED)
else()
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(SDL2 REQUIRED sdl2)
endif()
find_package(OpenGL REQUIRED)

set(IMGUI_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/imgui")

set(PROCESS_RUNNER_SOURCE src/process_runner_posix.cpp)
if(WIN32)
    set(PROCESS_RUNNER_SOURCE src/process_runner_win32.cpp)
endif()

add_executable(VideoComputePipelineUI
    src/main.cpp
    src/ui_app.cpp
    src/command_builder.cpp
    ${PROCESS_RUNNER_SOURCE}
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

target_compile_definitions(VideoComputePipelineUI PRIVATE SDL_MAIN_HANDLED)
if(WIN32)
    target_compile_definitions(VideoComputePipelineUI PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
endif()

if(TARGET SDL2::SDL2)
    target_link_libraries(VideoComputePipelineUI PRIVATE SDL2::SDL2)
elseif(TARGET SDL2::SDL2-static)
    target_link_libraries(VideoComputePipelineUI PRIVATE SDL2::SDL2-static)
elseif(DEFINED SDL2_LIBRARIES)
    target_include_directories(VideoComputePipelineUI PRIVATE ${SDL2_INCLUDE_DIRS})
    target_link_directories(VideoComputePipelineUI PRIVATE ${SDL2_LIBRARY_DIRS})
    target_link_libraries(VideoComputePipelineUI PRIVATE ${SDL2_LIBRARIES})
else()
    message(FATAL_ERROR "SDL2 was not found. Install SDL2 with vcpkg on Windows or libsdl2-dev on Linux.")
endif()

target_link_libraries(VideoComputePipelineUI PRIVATE OpenGL::GL)

if(WIN32)
    target_link_libraries(VideoComputePipelineUI PRIVATE shell32 ole32)
endif()

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
