#include "ui_app.h"

#include "path_utils.h"

#include "imgui.h"
#include "imgui_impl_opengl3.h"
#include "imgui_impl_sdl2.h"

#include <SDL.h>
#include <SDL_opengl.h>

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <vector>

namespace vcpui {

namespace {

bool input_text_string(const char *label, std::string &value) {
    char buffer[1024];
    std::snprintf(buffer, sizeof(buffer), "%s", value.c_str());
    if (ImGui::InputText(label, buffer, sizeof(buffer))) {
        value = buffer;
        return true;
    }
    return false;
}

bool combo_from_vector(const char *label, int &index, const std::vector<const char *> &items) {
    return ImGui::Combo(label, &index, items.data(), (int)items.size());
}

int index_of_value(const std::vector<const char *> &items, const std::string &value, int fallback) {
    for (int i = 0; i < (int)items.size(); ++i) {
        if (value == items[(size_t)i]) {
            return i;
        }
    }
    return fallback;
}

int clamp_int_min(int value, int minimum) {
    return value < minimum ? minimum : value;
}

float clamp_float(float value, float minimum, float maximum) {
    if (value < minimum) return minimum;
    if (value > maximum) return maximum;
    return value;
}

void text_pair(const char *label, const std::string &value) {
    ImGui::TextUnformatted(label);
    ImGui::SameLine(180.0f);
    ImGui::TextUnformatted(value.c_str());
}

void text_pair(const char *label, const char *value) {
    text_pair(label, std::string(value));
}

void help_section(const char *title, const char *body) {
    if (ImGui::CollapsingHeader(title, ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::TextWrapped("%s", body);
    }
}

std::string trim_copy(const std::string &value) {
    size_t begin = 0;
    size_t end = value.size();
    while (begin < end && (value[begin] == ' ' || value[begin] == '\t' || value[begin] == '\r' || value[begin] == '\n')) {
        ++begin;
    }
    while (end > begin && (value[end - 1u] == ' ' || value[end - 1u] == '\t' || value[end - 1u] == '\r' || value[end - 1u] == '\n')) {
        --end;
    }
    return value.substr(begin, end - begin);
}

std::string lower_copy(std::string value) {
    for (char &ch : value) {
        if (ch >= 'A' && ch <= 'Z') {
            ch = (char)(ch - 'A' + 'a');
        }
    }
    return value;
}

}  // namespace

int UiApp::run() {
    apply_preset(config_, Preset::PeopleDetectionAnnotatedVideo);
    normalize_default_paths();
    refresh_command();

    SDL_SetMainReady();
    if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_TIMER) != 0) {
        return 1;
    }

    SDL_GL_SetAttribute(SDL_GL_CONTEXT_FLAGS, 0);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 2);
    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
    SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);
    SDL_GL_SetAttribute(SDL_GL_STENCIL_SIZE, 8);

    SDL_Window *window = SDL_CreateWindow("VideoComputePipelineUI",
                                          SDL_WINDOWPOS_CENTERED,
                                          SDL_WINDOWPOS_CENTERED,
                                          1280,
                                          820,
                                          SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE | SDL_WINDOW_ALLOW_HIGHDPI);
    if (!window) {
        SDL_Quit();
        return 1;
    }

    SDL_GLContext glContext = SDL_GL_CreateContext(window);
    SDL_GL_MakeCurrent(window, glContext);
    SDL_GL_SetSwapInterval(1);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO &io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    ImGui::StyleColorsDark();

    ImGui_ImplSDL2_InitForOpenGL(window, glContext);
    ImGui_ImplOpenGL3_Init("#version 150");

    bool running = true;
    while (running) {
        SDL_Event event;
        while (SDL_PollEvent(&event)) {
            ImGui_ImplSDL2_ProcessEvent(&event);
            if (event.type == SDL_QUIT) {
                running = false;
            }
            if (event.type == SDL_WINDOWEVENT && event.window.event == SDL_WINDOWEVENT_CLOSE &&
                event.window.windowID == SDL_GetWindowID(window)) {
                running = false;
            }
        }

        runner_.poll();
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplSDL2_NewFrame();
        ImGui::NewFrame();

        render();

        ImGui::Render();
        int displayW = 0;
        int displayH = 0;
        SDL_GL_GetDrawableSize(window, &displayW, &displayH);
        glViewport(0, 0, displayW, displayH);
        glClearColor(0.08f, 0.09f, 0.10f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        SDL_GL_SwapWindow(window);
    }

    runner_.stop();
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplSDL2_Shutdown();
    ImGui::DestroyContext();
    SDL_GL_DeleteContext(glContext);
    SDL_DestroyWindow(window);
    SDL_Quit();
    return 0;
}

void UiApp::render() {
    ImGui::SetNextWindowPos(ImVec2(0, 0), ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImGui::GetIO().DisplaySize, ImGuiCond_Always);
    ImGui::Begin("VideoComputePipelineUI", nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove);

    if (ImGui::BeginTabBar("main-tabs")) {
        if (ImGui::BeginTabItem("Run Config")) {
            render_run_config_tab();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Monitor")) {
            render_monitor_tab();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Logs")) {
            render_logs_tab();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Help")) {
            render_help_tab();
            ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
    }
    ImGui::End();
}

void UiApp::render_run_config_tab() {
    int presetIndex = (int)config_.preset;
    if (combo_from_vector("Preset", presetIndex, preset_names())) {
        apply_preset(config_, (Preset)presetIndex);
        refresh_command();
    }

    if (input_text_string("Pipeline exe", config_.pipelineExePath)) mark_custom();
    render_tooltip("Path to VideoComputePipeline.exe. Defaults to ..\\VideoComputePipeline\\build-win\\bin\\VideoComputePipeline.exe.");
    if (input_text_string("Working directory", config_.workingDirectory)) mark_custom();
    render_tooltip("The subprocess working directory. Use ..\\VideoComputePipeline so relative data, model, output, and benchmark paths resolve correctly.");
    ImGui::Separator();

    int taskIndex = (int)config_.task;
    if (combo_from_vector("Task", taskIndex, task_names())) {
        config_.task = (Task)taskIndex;
        mark_custom();
    }
    const bool detectMode = config_.task == Task::Detect;
    const bool annotatedDetection = detectMode && config_.drawBoxes;

    int decoderIndex = (int)config_.decoder;
    ImGui::BeginDisabled(!detectMode);
    if (combo_from_vector("Decoder", decoderIndex, decoder_names())) {
        config_.decoder = (Decoder)decoderIndex;
        mark_custom();
    }
    ImGui::EndDisabled();
    render_tooltip("CPU decodes video frames in system RAM. NVDEC uses NVIDIA hardware decoding and can keep frames on GPU.");
    ImGui::BeginDisabled(!detectMode || config_.decoder != Decoder::Nvdec);
    if (ImGui::Checkbox("CPU decoder fallback", &config_.decoderFallbackCpu)) mark_custom();
    ImGui::EndDisabled();

    int encoderIndex = (int)config_.encoder;
    ImGui::BeginDisabled(detectMode && !annotatedDetection);
    if (combo_from_vector("Encoder", encoderIndex, encoder_names())) {
        config_.encoder = (Encoder)encoderIndex;
        mark_custom();
    }
    ImGui::EndDisabled();
    render_tooltip("h264_nvenc uses NVIDIA hardware encoding. Select none for CSV-only detection.");

    int precisionIndex = (int)config_.precision;
    ImGui::BeginDisabled(!detectMode);
    if (combo_from_vector("Precision", precisionIndex, precision_names())) {
        config_.precision = (Precision)precisionIndex;
        mark_custom();
    }
    ImGui::EndDisabled();
    render_tooltip("fp32 is safest and most compatible. fp16 can be faster only if the TensorRT engine and GPU support it.");

    int outputFormatIndex = (int)config_.outputFormat;
    ImGui::BeginDisabled(detectMode && !annotatedDetection);
    if (combo_from_vector("Output format", outputFormatIndex, output_format_names())) {
        config_.outputFormat = (OutputFormat)outputFormatIndex;
        mark_custom();
    }
    ImGui::EndDisabled();
    render_tooltip("MKV is better for viewing while the pipeline is still writing. MP4 may not be playable until finalized.");

    ImGui::BeginDisabled(detectMode);
    int processModeIndex = (int)config_.processMode;
    if (combo_from_vector("Filter mode", processModeIndex, process_mode_names())) {
        config_.processMode = (ProcessMode)processModeIndex;
        mark_custom();
    }
    int filterIndex = (int)config_.filter;
    if (combo_from_vector("Filter", filterIndex, filter_names())) {
        config_.filter = (Filter)filterIndex;
        mark_custom();
    }
    ImGui::EndDisabled();

    if (input_text_string("Input video", config_.inputVideoPath)) mark_custom();
    ImGui::BeginDisabled(!detectMode);
    if (input_text_string("TensorRT model", config_.modelPath)) mark_custom();
    if (input_text_string("Labels file", config_.labelsPath)) mark_custom();
    ImGui::EndDisabled();
    ImGui::BeginDisabled(detectMode && !annotatedDetection);
    if (input_text_string("Output video", config_.outputVideoPath)) mark_custom();
    ImGui::EndDisabled();
    ImGui::BeginDisabled(!detectMode);
    if (input_text_string("Detections CSV", config_.detectionsCsvPath)) mark_custom();
    ImGui::EndDisabled();
    if (input_text_string("Benchmark CSV", config_.benchmarkCsvPath)) mark_custom();

    ImGui::BeginDisabled(!detectMode);
    if (ImGui::Checkbox("Draw boxes", &config_.drawBoxes)) mark_custom();
    render_tooltip("Draw detection boxes and write annotated video. Leave disabled for CSV-only detection.");
    ImGui::EndDisabled();
    ImGui::BeginDisabled(!annotatedDetection);
    if (ImGui::InputInt("Box thickness", &config_.boxThickness)) mark_custom();
    if (ImGui::SliderFloat("Box confidence", &config_.boxConfidence, 0.0f, 1.0f, "%.2f")) mark_custom();
    ImGui::EndDisabled();
    ImGui::BeginDisabled(!detectMode);
    if (ImGui::SliderFloat("Confidence", &config_.confidence, 0.0f, 1.0f, "%.2f")) mark_custom();
    render_tooltip("Minimum detection confidence. Higher values reduce false detections but may miss weak objects.");
    if (ImGui::SliderFloat("IoU threshold", &config_.iouThreshold, 0.0f, 1.0f, "%.2f")) mark_custom();
    render_tooltip("Controls non-maximum suppression. Lower values remove overlapping boxes more aggressively.");
    if (ImGui::InputInt("Input size", &config_.inputSize)) mark_custom();
    render_tooltip("Model input resolution. YOLOv5s commonly uses 640.");

    refresh_labels_if_needed();
    ImGui::SeparatorText("Class filter");
    ImGui::Text("Selected: %zu", config_.selectedClassIds.size());
    ImGui::SameLine();
    if (ImGui::Button("Detect all classes")) {
        config_.selectedClassIds.clear();
        mark_custom();
    }
    ImGui::SameLine();
    if (ImGui::Button("Reload labels")) {
        loadedLabelsPath_.clear();
        refresh_labels_if_needed();
    }
    input_text_string("Search classes", classSearch_);
    render_tooltip("Leave the selection empty to detect every class. Select one or more labels to emit --class-ids.");
    if (classLabels_.empty()) {
        ImGui::TextColored(ImVec4(1.0f, 0.72f, 0.24f, 1.0f), "No labels loaded from the labels path.");
    } else {
        const std::string search = lower_copy(classSearch_);
        ImGui::BeginChild("class-filter-list", ImVec2(-1.0f, 150.0f), true);
        for (int classId = 0; classId < (int)classLabels_.size(); ++classId) {
            const std::string &label = classLabels_[(size_t)classId];
            if (!search.empty() && lower_copy(label).find(search) == std::string::npos) {
                continue;
            }
            bool selected = class_id_selected(classId);
            char itemLabel[256];
            std::snprintf(itemLabel, sizeof(itemLabel), "%d: %s", classId, label.c_str());
            if (ImGui::Checkbox(itemLabel, &selected)) {
                set_class_id_selected(classId, selected);
                mark_custom();
            }
        }
        ImGui::EndChild();
    }
    ImGui::EndDisabled();
    if (ImGui::InputInt("Max frames", &config_.maxFrames)) mark_custom();
    render_tooltip("0 means full video.");
    if (ImGui::InputInt("Progress interval", &config_.progressInterval)) mark_custom();
    int ffmpegLogIndex = index_of_value(ffmpeg_log_level_names(), config_.ffmpegLogLevel, 1);
    if (combo_from_vector("FFmpeg log level", ffmpegLogIndex, ffmpeg_log_level_names())) {
        config_.ffmpegLogLevel = ffmpeg_log_level_names()[(size_t)ffmpegLogIndex];
        mark_custom();
    }
    if (ImGui::Checkbox("Benchmark enabled", &config_.benchmarkEnabled)) mark_custom();

    config_.boxThickness = clamp_int_min(config_.boxThickness, 1);
    config_.inputSize = clamp_int_min(config_.inputSize, 1);
    config_.maxFrames = clamp_int_min(config_.maxFrames, 0);
    config_.progressInterval = clamp_int_min(config_.progressInterval, 0);
    config_.confidence = clamp_float(config_.confidence, 0.0f, 1.0f);
    config_.iouThreshold = clamp_float(config_.iouThreshold, 0.0f, 1.0f);
    config_.boxConfidence = clamp_float(config_.boxConfidence, 0.0f, 1.0f);
    refresh_command();

    ImGui::Separator();
    ImGui::TextUnformatted("Generated command");
    std::vector<char> previewBuffer(command_.preview.begin(), command_.preview.end());
    previewBuffer.push_back('\0');
    ImGui::InputTextMultiline("##command-preview",
                              previewBuffer.data(),
                              previewBuffer.size(),
                              ImVec2(-1.0f, 92.0f),
                              ImGuiInputTextFlags_ReadOnly);
    if (ImGui::Button("Copy command")) {
        SDL_SetClipboardText(command_.preview.c_str());
    }
    ImGui::SameLine();
    if (ImGui::Button("Run")) {
        start_pipeline();
    }
    ImGui::SameLine();
    if (ImGui::Button("Stop")) {
        stop_pipeline();
    }

    if (validationEnabled_) {
        const auto issues = validate_config(config_);
        for (const auto &issue : issues) {
            ImGui::TextColored(ImVec4(1.0f, 0.72f, 0.24f, 1.0f), "%s", issue.c_str());
        }
    }
}

void UiApp::render_monitor_tab() {
    RunProgress progress = runner_.progress_snapshot();
    update_elapsed_progress(progress, runner_.elapsed_seconds());
    if (config_.task == Task::Detect && config_.drawBoxes && !config_.outputVideoPath.empty()) {
        const std::string outputPath = resolve_path(config_.workingDirectory, config_.outputVideoPath);
        progress.outputBytes = file_size_bytes(outputPath);
    }
    const ProcessStatus status = runner_.status();

    text_pair("Status", status_to_string(status));
    text_pair("Elapsed", format_duration(progress.elapsedSeconds));
    text_pair("ETA", progress.etaSeconds > 0.0 ? format_duration(progress.etaSeconds) : std::string("--"));
    text_pair("Current FPS", progress.fps > 0.0 ? std::to_string(progress.fps) : std::string("--"));
    text_pair("Speed", progress.speed > 0.0 ? std::to_string(progress.speed) : std::string("--"));
    text_pair("Frames", std::to_string(progress.framesProcessed) + (progress.totalFrames > 0 ? " / " + std::to_string(progress.totalFrames) : ""));
    text_pair("Detections", progress.totalDetections > 0 ? std::to_string(progress.totalDetections) : std::string("--"));
    text_pair("Output size", progress.outputBytes > 0 ? format_bytes(progress.outputBytes) : std::string("--"));

    if (progress.progress > 0.0) {
        ImGui::ProgressBar((float)progress.progress, ImVec2(-1.0f, 0.0f));
    } else {
        ImGui::ProgressBar(0.0f, ImVec2(-1.0f, 0.0f), "progress unavailable");
    }
    ImGui::Separator();
    ImGui::TextWrapped("Last status: %s", progress.lastStatusLine.empty() ? "--" : progress.lastStatusLine.c_str());
}

void UiApp::render_logs_tab() {
    input_text_string("Filter", logFilter_);
    ImGui::SameLine();
    ImGui::Checkbox("Auto-scroll", &autoScroll_);
    ImGui::SameLine();
    if (ImGui::Button("Clear logs")) {
        runner_.clear_logs();
    }
    ImGui::SameLine();
    if (ImGui::Button("Save logs")) {
        std::ofstream out("VideoComputePipelineUI_logs.txt");
        for (const auto &line : runner_.logs_snapshot()) {
            out << line.text << '\n';
        }
    }

    ImGui::BeginChild("log-window", ImVec2(0, 0), true, ImGuiWindowFlags_HorizontalScrollbar);
    for (const auto &line : runner_.logs_snapshot()) {
        if (!logFilter_.empty() && line.text.find(logFilter_) == std::string::npos) {
            continue;
        }
        const bool important = line.isError ||
                               line.text.find("ERROR") != std::string::npos ||
                               line.text.find("failed") != std::string::npos ||
                               line.text.find("WARN") != std::string::npos ||
                               line.text.find("TensorRT") != std::string::npos;
        if (important) {
            ImGui::TextColored(ImVec4(1.0f, 0.35f, 0.30f, 1.0f), "%s", line.text.c_str());
        } else {
            ImGui::TextUnformatted(line.text.c_str());
        }
    }
    if (autoScroll_ && ImGui::GetScrollY() >= ImGui::GetScrollMaxY()) {
        ImGui::SetScrollHereY(1.0f);
    }
    ImGui::EndChild();
}

void UiApp::render_help_tab() {
    help_section("What VideoComputePipeline does",
                 "VideoComputePipeline reads video through FFmpeg libraries, runs CPU/OpenCL filters or CUDA/TensorRT detection, writes CSV benchmark data, and can optionally write annotated video.");
    help_section("Filter vs detection",
                 "Filter mode changes pixels and writes a processed video. Detection mode is CSV-only by default and writes detections; annotated video is enabled with draw boxes and an output path.");
    help_section("CPU decoder vs NVDEC",
                 "CPU decoding produces system-memory frames. NVDEC uses NVIDIA hardware decode and is the preferred high-throughput path for GPU-resident detection and annotation.");
    help_section("Encoder choices",
                 "Use encoder none for CSV-only detection. Use h264_nvenc for low-CPU annotated output. CPU encoders are mainly for filter mode.");
    help_section("TensorRT, models, and labels",
                 "Detection needs a TensorRT engine file and a labels file. The engine precision must be compatible with the runtime precision and GPU.");
    help_section("Confidence, IoU, and input size",
                 "Confidence filters weak detections. IoU controls overlap suppression. YOLOv5s commonly uses input size 640; larger values cost more FPS.");
    help_section("Class filtering",
                 "By default detection keeps every class the model can output. Select one or more labels in the Run Config tab to emit class IDs and keep only those classes in CSV and annotated output.");
    help_section("MKV vs MP4",
                 "MKV is more tolerant while files are being written or if a run is interrupted. MP4 is often not playable until the trailer is finalized.");
    help_section("Common errors",
                 "Check missing DLLs, missing TensorRT engine, missing labels, wrong working directory, unavailable NVDEC/NVENC, TensorRT engine mismatch, and output files open in another program.");
}

void UiApp::render_tooltip(const char *text) {
    if (ImGui::IsItemHovered()) {
        ImGui::BeginTooltip();
        ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
        ImGui::TextUnformatted(text);
        ImGui::PopTextWrapPos();
        ImGui::EndTooltip();
    }
}

void UiApp::mark_custom() {
    config_.preset = Preset::Custom;
}

void UiApp::start_pipeline() {
    refresh_command();
    runner_.start(command_.win32CommandLine, utf8_to_wide(config_.workingDirectory));
}

void UiApp::stop_pipeline() {
    runner_.stop();
}

void UiApp::refresh_command() {
    command_ = build_command(config_);
}

void UiApp::normalize_default_paths() {
    namespace fs = std::filesystem;

    fs::path cursor = fs::u8path(executable_directory());
    while (!cursor.empty()) {
        const fs::path pipelineDir = cursor / "VideoComputePipeline";
        const fs::path uiDir = cursor / "VideoComputePipelineUI";
        if (fs::is_directory(pipelineDir) && fs::is_directory(uiDir)) {
            config_.workingDirectory = pipelineDir.u8string();

            const fs::path candidates[] = {
                pipelineDir / "build-msvc-hw" / "bin" / "VideoComputePipeline.exe",
                pipelineDir / "build-msvc" / "bin" / "VideoComputePipeline.exe",
                pipelineDir / "build-win" / "bin" / "VideoComputePipeline.exe",
                pipelineDir / "build-vs" / "Release" / "VideoComputePipeline.exe",
            };
            for (const fs::path &candidate : candidates) {
                if (fs::is_regular_file(candidate)) {
                    config_.pipelineExePath = candidate.u8string();
                    return;
                }
            }
            return;
        }

        const fs::path parent = cursor.parent_path();
        if (parent == cursor) {
            break;
        }
        cursor = parent;
    }
}

void UiApp::refresh_labels_if_needed() {
    const std::string labelPath = resolve_path(config_.workingDirectory, config_.labelsPath);
    if (labelPath == loadedLabelsPath_) {
        return;
    }

    loadedLabelsPath_ = labelPath;
    classLabels_.clear();

    std::ifstream file(labelPath);
    std::string line;
    while (std::getline(file, line)) {
        const std::string label = trim_copy(line);
        if (!label.empty()) {
            classLabels_.push_back(label);
        }
    }

    config_.selectedClassIds.erase(
        std::remove_if(config_.selectedClassIds.begin(),
                       config_.selectedClassIds.end(),
                       [this](int id) { return id < 0 || id >= (int)classLabels_.size(); }),
        config_.selectedClassIds.end());
}

bool UiApp::class_id_selected(int class_id) const {
    return std::find(config_.selectedClassIds.begin(), config_.selectedClassIds.end(), class_id) != config_.selectedClassIds.end();
}

void UiApp::set_class_id_selected(int class_id, bool selected) {
    if (selected) {
        if (!class_id_selected(class_id)) {
            config_.selectedClassIds.push_back(class_id);
        }
    } else {
        config_.selectedClassIds.erase(std::remove(config_.selectedClassIds.begin(), config_.selectedClassIds.end(), class_id),
                                       config_.selectedClassIds.end());
    }
}

}  // namespace vcpui
