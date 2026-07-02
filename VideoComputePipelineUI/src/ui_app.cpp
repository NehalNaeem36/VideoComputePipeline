/*
 * UI app module: owns the Dear ImGui interface, user-editable pipeline config,
 * command preview, monitor, logs, and help tabs. It coordinates command_builder,
 * process_runner, progress_parser, and path utilities without linking pipeline
 * internals.
 */
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

std::string format_fixed(double value, int precision = 3) {
    std::ostringstream out;
    out.setf(std::ios::fixed);
    out.precision(precision);
    out << value;
    return out.str();
}

const char *bool_text(bool value) {
    return value ? "true" : "false";
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

std::vector<std::string> missing_ffmpeg_runtime_dlls(const PipelineRunConfig &config) {
    const std::string exeDir = parent_directory(config.pipelineExePath);
    const char *dlls[] = {
        "avcodec-62.dll",
        "avformat-62.dll",
        "avutil-60.dll",
        "swscale-9.dll",
        "swresample-6.dll",
        "avdevice-62.dll",
        "avfilter-11.dll",
    };
    std::vector<std::string> missing;
    for (const char *dll : dlls) {
        if (!file_exists(resolve_path(exeDir, dll))) {
            missing.emplace_back(dll);
        }
    }
    return missing;
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
    ImGui::SeparatorText("Paths");
    int presetIndex = (int)config_.preset;
    if (combo_from_vector("Preset", presetIndex, preset_names())) {
        apply_preset(config_, (Preset)presetIndex);
        refresh_command();
    }
    render_tooltip("Preset command profiles. Custom preserves your manual edits.");

    if (input_text_string("Pipeline exe", config_.pipelineExePath)) mark_custom();
    render_tooltip("Path to VideoComputePipeline.exe. Use build-win-cuda12 for CUDA filters, TensorRT, ONNX Runtime, NVDEC, and NVENC. CPU-only builds cannot run --mode gpu.");
    if (input_text_string("Working directory", config_.workingDirectory)) mark_custom();
    render_tooltip("Subprocess working directory. Relative input, model, label, output, benchmark, and detection paths are resolved from the VideoComputePipeline folder.");
    const std::vector<std::string> missingFfmpegDlls = missing_ffmpeg_runtime_dlls(config_);
    if (missingFfmpegDlls.empty()) {
        ImGui::TextColored(ImVec4(0.40f, 0.90f, 0.55f, 1.0f), "FFmpeg runtime: vcpkg DLLs found beside selected executable");
    } else {
        ImGui::TextColored(ImVec4(1.0f, 0.72f, 0.24f, 1.0f), "FFmpeg runtime: %zu expected DLL(s) missing beside selected executable", missingFfmpegDlls.size());
    }
    render_tooltip("MSVC/CUDA runs should load vcpkg FFmpeg DLLs from the same folder as VideoComputePipeline.exe. This avoids accidentally loading MSYS2 or another FFmpeg from PATH.");

    ImGui::SeparatorText("Task");
    int taskIndex = (int)config_.task;
    if (combo_from_vector("Task", taskIndex, task_names())) {
        config_.task = (Task)taskIndex;
        mark_custom();
    }
    render_tooltip("Filter modifies video pixels and writes an output video. Detect writes detections CSV by default and can optionally write annotated video.");
    const bool detectMode = config_.task == Task::Detect;
    const bool annotatedDetection = detectMode && config_.drawBoxes;
    const bool profileOnly = detectMode && config_.profileHardwareOnly;

    ImGui::SeparatorText("Video I/O");
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
    render_tooltip("Allows CPU decode if NVDEC cannot open the input. Disable this when you want NVDEC failures to be visible.");

    int runtimeIndex = (int)config_.runtime;
    ImGui::BeginDisabled(!detectMode);
    if (combo_from_vector("Runtime", runtimeIndex, runtime_names())) {
        config_.runtime = (Runtime)runtimeIndex;
        mark_custom();
    }
    ImGui::EndDisabled();
    render_tooltip("Inference runtime passed as --runtime. ONNX Runtime uses .onnx models. TensorRT uses .engine/.plan models. TorchScript requires a LibTorch-enabled build.");

    int encoderIndex = (int)config_.encoder;
    ImGui::BeginDisabled(profileOnly || (detectMode && !annotatedDetection));
    if (combo_from_vector("Encoder", encoderIndex, encoder_names())) {
        config_.encoder = (Encoder)encoderIndex;
        mark_custom();
    }
    ImGui::EndDisabled();
    render_tooltip("h264_nvenc uses NVIDIA hardware encoding for annotated detection. Select none for CSV-only detection.");

    int precisionIndex = (int)config_.precision;
    ImGui::BeginDisabled(!detectMode);
    if (combo_from_vector("Precision", precisionIndex, precision_names())) {
        config_.precision = (Precision)precisionIndex;
        mark_custom();
    }
    ImGui::EndDisabled();
    render_tooltip("Runtime precision label. The actual model input tensor type still determines whether FP32 or FP16 preprocessing is used.");

    int outputFormatIndex = (int)config_.outputFormat;
    ImGui::BeginDisabled(profileOnly || (detectMode && !annotatedDetection));
    if (combo_from_vector("Output format", outputFormatIndex, output_format_names())) {
        config_.outputFormat = (OutputFormat)outputFormatIndex;
        mark_custom();
    }
    ImGui::EndDisabled();
    render_tooltip("MKV is recommended for long annotated hardware-video runs. MP4 may not be playable until the trailer is finalized.");

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

    ImGui::BeginDisabled(profileOnly);
    if (input_text_string("Input video", config_.inputVideoPath)) mark_custom();
    render_tooltip("Input video path relative to the working directory unless an absolute path is provided.");
    ImGui::EndDisabled();

    ImGui::SeparatorText("Detection Model");
    ImGui::BeginDisabled(!detectMode);
    if (input_text_string("Model", config_.modelPath)) mark_custom();
    render_tooltip("Model file for the selected runtime: .engine/.plan for TensorRT, .onnx for ONNX Runtime, or TorchScript .pt/.ts when LibTorch is enabled. It must match model family, input size, and YOLO output layout.");
    if (input_text_string("Labels file", config_.labelsPath)) mark_custom();
    render_tooltip("Class labels file. One label per line; class IDs are the line numbers starting at 0.");
    ImGui::EndDisabled();
    ImGui::BeginDisabled(profileOnly || (detectMode && !annotatedDetection));
    if (input_text_string("Output video", config_.outputVideoPath)) mark_custom();
    ImGui::EndDisabled();
    render_tooltip("Annotated output video path. Detection CSV-only runs do not need this.");
    ImGui::BeginDisabled(!detectMode);
    if (input_text_string("Detections CSV", config_.detectionsCsvPath)) mark_custom();
    ImGui::EndDisabled();
    render_tooltip("Detection rows are streamed here. CSV-only and annotated detection both write this file.");
    ImGui::BeginDisabled(profileOnly);
    if (input_text_string("Benchmark CSV", config_.benchmarkCsvPath)) mark_custom();
    ImGui::EndDisabled();
    render_tooltip("Per-frame benchmark output. Disable benchmark if you only need logs or hardware profiling.");

    ImGui::BeginDisabled(!detectMode);
    if (ImGui::Checkbox("Draw boxes", &config_.drawBoxes)) mark_custom();
    render_tooltip("Draw detection boxes and write annotated video. Leave disabled for CSV-only detection.");
    ImGui::EndDisabled();
    ImGui::BeginDisabled(!annotatedDetection);
    if (ImGui::InputInt("Box thickness", &config_.boxThickness)) mark_custom();
    render_tooltip("Bounding box border thickness in pixels for annotated output.");
    if (ImGui::SliderFloat("Box confidence", &config_.boxConfidence, 0.0f, 1.0f, "%.2f")) mark_custom();
    render_tooltip("Minimum confidence for drawing boxes. This can be higher than the detection CSV threshold.");
    ImGui::EndDisabled();
    ImGui::BeginDisabled(!detectMode);
    if (ImGui::SliderFloat("Confidence", &config_.confidence, 0.0f, 1.0f, "%.2f")) mark_custom();
    render_tooltip("Minimum detection confidence. Lower values find more objects; higher values reduce false positives but may miss weak objects.");
    if (ImGui::SliderFloat("IoU threshold", &config_.iouThreshold, 0.0f, 1.0f, "%.2f")) mark_custom();
    render_tooltip("Controls non-maximum suppression. Lower values remove overlapping boxes more aggressively.");
    if (ImGui::InputInt("Input size", &config_.inputSize)) mark_custom();
    render_tooltip("YOLO network input resolution. The full frame is letterboxed into this size; it is not a crop. YOLOv5s commonly uses 640.");

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
    render_tooltip("Leave empty to detect every class. Select one or more labels to emit --class-ids and restrict CSV and annotated output.");
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

    ImGui::SeparatorText("Staged Execution");
    ImGui::BeginDisabled(!detectMode);
    if (ImGui::Checkbox("Auto tune", &config_.autoTune)) mark_custom();
    render_tooltip("Lets the pipeline choose physical schedule batch, in-flight batch pool size, and inference lane settings from the video, GPU, and selected model/runtime limits.");
    if (ImGui::Checkbox("Profile hardware only", &config_.profileHardwareOnly)) mark_custom();
    render_tooltip("Prints CUDA/VRAM capability information without requiring a full detection run.");

    int batchModeIndex = (int)config_.batchSizeMode;
    if (combo_from_vector("Schedule batch mode", batchModeIndex, auto_int_mode_names())) {
        config_.batchSizeMode = (AutoIntMode)batchModeIndex;
        mark_custom();
    }
    render_tooltip("Auto lets the planner choose. Manual sends the selected physical scheduling batch size.");
    ImGui::BeginDisabled(config_.batchSizeMode == AutoIntMode::Auto);
    if (ImGui::InputInt("Schedule batch size", &config_.batchSize)) mark_custom();
    ImGui::EndDisabled();
    render_tooltip("Frames per reusable FrameBatch. This controls how many decoded NV12 frames move together through the staged pipeline. If backend batch is 1, frames still run through single-frame inference lanes.");

    int inflightModeIndex = (int)config_.inflightBatchesMode;
    if (combo_from_vector("In-flight batches mode", inflightModeIndex, auto_int_mode_names())) {
        config_.inflightBatchesMode = (AutoIntMode)inflightModeIndex;
        mark_custom();
    }
    render_tooltip("Auto lets the planner choose how many reusable FrameBatch objects can be active across stages.");
    ImGui::BeginDisabled(config_.inflightBatchesMode == AutoIntMode::Auto);
    if (ImGui::InputInt("In-flight batches", &config_.inflightBatches)) mark_custom();
    ImGui::EndDisabled();
    render_tooltip("Number of reusable FrameBatch objects in the free/decoded/completed queues. Higher values let NVDEC decode N+2 while inference handles N+1 and output/NVENC writes N.");

    if (ImGui::InputFloat("Target FPS", &config_.targetFps, 1.0f, 10.0f, "%.1f")) mark_custom();
    render_tooltip("Tuning goal used by the planner. It is not a guaranteed output rate.");
    if (ImGui::SliderFloat("VRAM budget ratio", &config_.vramBudgetRatio, 0.05f, 0.90f, "%.3f")) mark_custom();
    render_tooltip("Fraction of total GPU memory the planner may use for active batches and buffers.");
    if (ImGui::InputInt("VRAM reserve MB", &config_.vramReserveMb)) mark_custom();
    render_tooltip("GPU memory kept free for the OS, driver, desktop, and other GPU work. 0 lets the pipeline compute a reserve.");

    int overlapIndex = (int)config_.pipelineOverlap;
    if (combo_from_vector("Pipeline overlap", overlapIndex, feature_mode_names())) {
        config_.pipelineOverlap = (FeatureMode)overlapIndex;
        mark_custom();
    }
    render_tooltip("Enables real stage overlap: decoder thread fills batches, inference worker(s) process decoded batches, and the output stage writes detections/overlay/NVENC in frame order.");
    int parallelIndex = (int)config_.parallelInference;
    if (combo_from_vector("Parallel inference", parallelIndex, feature_mode_names())) {
        config_.parallelInference = (FeatureMode)parallelIndex;
        mark_custom();
    }
    render_tooltip("Uses multiple inference workers/lanes when supported. Each lane owns its backend resources, so static batch-1 models can still process different frames concurrently.");

    int contextsModeIndex = (int)config_.inferenceContextsMode;
    if (combo_from_vector("Inference contexts mode", contextsModeIndex, auto_int_mode_names())) {
        config_.inferenceContextsMode = (AutoIntMode)contextsModeIndex;
        mark_custom();
    }
    render_tooltip("Auto lets the planner choose inference worker count. Manual requests a specific number of lanes.");
    ImGui::BeginDisabled(config_.inferenceContextsMode == AutoIntMode::Auto);
    if (ImGui::InputInt("Inference lanes", &config_.inferenceContexts)) mark_custom();
    ImGui::EndDisabled();
    render_tooltip("Requested inference workers. Use this to benchmark multiple model contexts/sessions even when true backend batching is unavailable.");
    ImGui::EndDisabled();

    ImGui::SeparatorText("Runtime");
    if (ImGui::InputInt("Max frames", &config_.maxFrames)) mark_custom();
    render_tooltip("0 means full video.");
    if (ImGui::InputInt("Progress interval", &config_.progressInterval)) mark_custom();
    render_tooltip("How many completed frames between progress log lines. Smaller values update the UI more often but produce more logs.");
    int ffmpegLogIndex = index_of_value(ffmpeg_log_level_names(), config_.ffmpegLogLevel, 1);
    if (combo_from_vector("FFmpeg log level", ffmpegLogIndex, ffmpeg_log_level_names())) {
        config_.ffmpegLogLevel = ffmpeg_log_level_names()[(size_t)ffmpegLogIndex];
        mark_custom();
    }
    render_tooltip("Use error for noisy files. Use debug only when diagnosing FFmpeg decode/encode problems.");
    ImGui::BeginDisabled(profileOnly);
    if (ImGui::Checkbox("Benchmark enabled", &config_.benchmarkEnabled)) mark_custom();
    ImGui::EndDisabled();

    config_.boxThickness = clamp_int_min(config_.boxThickness, 1);
    config_.batchSize = clamp_int_min(config_.batchSize, 1);
    config_.inflightBatches = clamp_int_min(config_.inflightBatches, 1);
    config_.inferenceContexts = clamp_int_min(config_.inferenceContexts, 1);
    config_.vramReserveMb = clamp_int_min(config_.vramReserveMb, 0);
    config_.inputSize = clamp_int_min(config_.inputSize, 1);
    config_.maxFrames = clamp_int_min(config_.maxFrames, 0);
    config_.progressInterval = clamp_int_min(config_.progressInterval, 0);
    config_.confidence = clamp_float(config_.confidence, 0.0f, 1.0f);
    config_.iouThreshold = clamp_float(config_.iouThreshold, 0.0f, 1.0f);
    config_.boxConfidence = clamp_float(config_.boxConfidence, 0.0f, 1.0f);
    config_.targetFps = clamp_float(config_.targetFps, 0.0f, 10000.0f);
    config_.vramBudgetRatio = clamp_float(config_.vramBudgetRatio, 0.05f, 0.90f);
    refresh_command();

    ImGui::SeparatorText("Generated Command");
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
    if (progress.totalFrames <= 0 && config_.maxFrames > 0) {
        progress.totalFrames = config_.maxFrames;
        if (progress.framesProcessed > 0) {
            progress.progress = clamp_float((float)progress.framesProcessed / (float)progress.totalFrames, 0.0f, 1.0f);
        }
    }
    update_elapsed_progress(progress, runner_.elapsed_seconds());
    if (config_.task == Task::Detect && config_.drawBoxes && !config_.outputVideoPath.empty()) {
        const std::string outputPath = resolve_path(config_.workingDirectory, config_.outputVideoPath);
        progress.outputBytes = file_size_bytes(outputPath);
    }
    const ProcessStatus status = runner_.status();

    text_pair("Status", status_to_string(status));
    text_pair("Elapsed", format_duration(progress.elapsedSeconds));
    text_pair("ETA", progress.etaSeconds > 0.0 ? format_duration(progress.etaSeconds) : std::string("--"));
    text_pair("Estimated total", progress.estimatedTotalSeconds > 0.0 ? format_duration(progress.estimatedTotalSeconds) : std::string("--"));
    text_pair("Current FPS", progress.fps > 0.0 ? std::to_string(progress.fps) : std::string("--"));
    text_pair("Speed", progress.speed > 0.0 ? std::to_string(progress.speed) : std::string("--"));
    text_pair("Frames", std::to_string(progress.framesProcessed) + (progress.totalFrames > 0 ? " / " + std::to_string(progress.totalFrames) : ""));
    text_pair("Detections", progress.totalDetections > 0 ? std::to_string(progress.totalDetections) : std::string("--"));
    text_pair("Output size", progress.outputBytes > 0 ? format_bytes(progress.outputBytes) : std::string("--"));
    text_pair("Execution mode", progress.executionMode >= 0 ? std::to_string(progress.executionMode) : std::string("--"));
    const int shownScheduleBatch = progress.scheduleBatchSize > 0 ? progress.scheduleBatchSize : progress.batchSize;
    text_pair("Schedule batch", shownScheduleBatch > 0 ? std::to_string(shownScheduleBatch) + " frame(s)" : std::string("--"));
    text_pair("Backend batch", progress.backendBatchSize > 0 ? std::to_string(progress.backendBatchSize) + " frame(s)" : std::string("--"));
    text_pair("In-flight batch pool", progress.inflightBatches > 0 ? std::to_string(progress.inflightBatches) + " batch object(s)" : std::string("--"));
    text_pair("Active frame capacity", progress.activeFrameCapacity > 0 ? std::to_string(progress.activeFrameCapacity) : (progress.totalActiveFrames > 0 ? std::to_string(progress.totalActiveFrames) : std::string("--")));
    text_pair("Inference workers", progress.inferenceLaneCount > 0 ? std::to_string(progress.inferenceLaneCount) : (progress.inferenceContextCount > 0 ? std::to_string(progress.inferenceContextCount) : std::string("--")));
    text_pair("Stage overlap", progress.executionMode >= 0 || progress.inflightBatches > 0 ? bool_text(progress.pipelineOverlapEnabled) : "--");
    text_pair("Parallel inference", progress.executionMode >= 0 || progress.inferenceLaneCount > 0 ? bool_text(progress.parallelInferenceEnabled) : "--");
    text_pair("Upload/download batch", progress.executionMode >= 0 ? std::to_string(progress.framesPerUploadBatch) + " / " + std::to_string(progress.framesPerDownloadBatch) : std::string("--"));
    text_pair("VRAM plan", progress.vramBudgetMb > 0.0 ? format_fixed(progress.vramBudgetMb) + " MB budget, " + format_fixed(progress.estimatedBatchMb) + " MB batch" : std::string("--"));
    text_pair("Unused VRAM budget", progress.unusedVramBudgetMb > 0.0 ? format_fixed(progress.unusedVramBudgetMb) + " MB" : std::string("--"));
    if (shownScheduleBatch > 0 && progress.inflightBatches > 0) {
        ImGui::TextWrapped("Topology: decoder fills batch N+2 while inference handles N+1 and output/NVENC writes N.");
    }

    if (progress.progress > 0.0) {
        ImGui::ProgressBar((float)progress.progress, ImVec2(-1.0f, 0.0f));
    } else {
        ImGui::ProgressBar(0.0f, ImVec2(-1.0f, 0.0f), "progress unavailable");
    }
    ImGui::Separator();
    if (!progress.fallbackReason.empty()) {
        ImGui::TextWrapped("Plan reason: %s", progress.fallbackReason.c_str());
    }
    ImGui::TextWrapped("Last status: %s", progress.lastStatusLine.empty() ? "--" : progress.lastStatusLine.c_str());
}

void UiApp::render_logs_tab() {
    input_text_string("Filter", logFilter_);
    ImGui::SameLine();
    ImGui::Checkbox("Auto-scroll", &autoScroll_);
    ImGui::SameLine();
    ImGui::Checkbox("Errors", &logShowErrors_);
    ImGui::SameLine();
    ImGui::Checkbox("Warnings", &logShowWarnings_);
    ImGui::SameLine();
    ImGui::Checkbox("TensorRT", &logShowTensorRt_);
    ImGui::SameLine();
    ImGui::Checkbox("Execution plan", &logShowExecutionPlan_);
    ImGui::SameLine();
    if (ImGui::Button("Clear logs")) {
        runner_.clear_logs();
    }
    ImGui::SameLine();
    if (ImGui::Button("Save logs")) {
        const std::string logPath = resolve_path(executable_directory(), "VideoComputePipelineUI_logs.txt");
        const auto lines = runner_.logs_snapshot();
        std::ofstream out(std::filesystem::u8path(logPath), std::ios::out | std::ios::trunc);
        if (!out) {
            logSaveStatus_ = "Failed to save logs: " + logPath;
        } else {
            for (const auto &line : lines) {
                out << line.text << '\n';
            }
            out.close();
            logSaveStatus_ = "Saved " + std::to_string(lines.size()) + " log lines to " + logPath;
        }
    }
    if (!logSaveStatus_.empty()) {
        ImGui::TextWrapped("%s", logSaveStatus_.c_str());
    }

    ImGui::BeginChild("log-window", ImVec2(0, 0), true, ImGuiWindowFlags_HorizontalScrollbar);
    for (const auto &line : runner_.logs_snapshot()) {
        if (!logFilter_.empty() && line.text.find(logFilter_) == std::string::npos) {
            continue;
        }
        const std::string lower = lower_copy(line.text);
        const bool isError = line.isError || lower.find("error") != std::string::npos || lower.find("failed") != std::string::npos;
        const bool isWarning = lower.find("warn") != std::string::npos;
        const bool isTensorRt = line.text.find("TensorRT") != std::string::npos;
        const bool isExecutionPlan = lower.find("execution plan") != std::string::npos ||
                                     lower.find("hardware profile") != std::string::npos ||
                                     lower.find("fallback") != std::string::npos ||
                                     lower.find("batch_size") != std::string::npos ||
                                     lower.find("backend_batch") != std::string::npos ||
                                     lower.find("inference_lane") != std::string::npos ||
                                     lower.find("inflight_batches") != std::string::npos ||
                                     lower.find("inference_context") != std::string::npos;
        if ((!logShowErrors_ && isError) ||
            (!logShowWarnings_ && isWarning) ||
            (!logShowTensorRt_ && isTensorRt) ||
            (!logShowExecutionPlan_ && isExecutionPlan)) {
            continue;
        }
        if (isError) {
            ImGui::TextColored(ImVec4(1.0f, 0.35f, 0.30f, 1.0f), "%s", line.text.c_str());
        } else if (isWarning) {
            ImGui::TextColored(ImVec4(1.0f, 0.72f, 0.24f, 1.0f), "%s", line.text.c_str());
        } else if (isTensorRt || isExecutionPlan) {
            ImGui::TextColored(ImVec4(0.45f, 0.78f, 1.0f, 1.0f), "%s", line.text.c_str());
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
    help_section("Pipeline modes",
                 "Filter mode changes pixels and writes a processed video. Detection mode runs the selected inference runtime, writes detections CSV by default, and writes annotated video only when Draw boxes and Output video are enabled.");
    help_section("CSV-only detection",
                 "CSV-only detection uses the model and labels paths, writes detections and benchmark CSV files, and does not need an encoder or output video path. This is the simplest mode for analytics-only inference.");
    help_section("Annotated detection",
                 "Annotated detection requires Draw boxes, an output path, and normally h264_nvenc. MKV is recommended for long hardware-video runs because it is more tolerant while writing and after interrupted runs.");
    help_section("CPU decode vs NVDEC",
                 "CPU decoding produces system-memory NV12 frames and uploads them to CUDA. NVDEC keeps decoded frames GPU-resident, avoiding raw-frame upload and enabling GPU overlay plus NVENC output.");
    help_section("FFmpeg runtime DLLs",
                 "The MSVC/CUDA build must use the vcpkg FFmpeg DLLs copied beside VideoComputePipeline.exe. The UI validates avcodec, avformat, avutil, swscale, swresample, avdevice, and avfilter DLLs in the executable directory so launches do not depend on PATH or accidentally load MSYS2 FFmpeg.");
    help_section("Inference runtimes",
                 "TensorRT loads .engine or .plan files and supports multi-context execution. ONNX Runtime loads .onnx files, uses CUDA I/O binding, and can use separate lane sessions for forced parallel single-frame inference. TorchScript is listed for future LibTorch builds and is unavailable unless the pipeline is rebuilt with ENABLE_LIBTORCH=ON.");
    help_section("Execution plan modes",
                 "Mode 0 is single-frame compatibility. Mode 1 enables batched transfers around sequential inference. Mode 2 overlaps pipeline stages while inference remains sequential. Mode 3 overlaps stages and uses parallel inference contexts when the runtime and GPU allow it.");
    help_section("Staged hardware pipeline",
                 "NVDEC hardware detection now runs as separate stages. A decoder thread fills reusable FrameBatch objects, inference worker(s) consume decoded batches, and the output stage writes detections, draws boxes, runs NVENC, records benchmarks, releases hardware frames, and returns each batch to the pool.");
    help_section("Batch size vs backend batch",
                 "Schedule batch is frames per reusable FrameBatch. Backend batch is how many frames the model runtime actually accepts in one inference call. A static batch-1 model can still benefit from schedule batches because decode, inference, and output are overlapped around it.");
    help_section("In-flight batches",
                 "In-flight batches is the number of reusable FrameBatch objects allocated for the hardware pipeline. Three in-flight batches allows decode batch N+2, inference batch N+1, and output batch N to be active at the same time without reallocating frame storage.");
    help_section("Parallel inference",
                 "Parallel inference means multiple inference workers, each with its own runtime-owned context or session. TensorRT uses separate execution contexts; ONNX Runtime uses separate lane sessions. This is different from true model batching and works with static batch-1 models when the backend can create multiple lanes.");
    help_section("VRAM budgeting",
                 "Auto tune estimates video frame memory, model/backend memory, active batch memory, and reserved GPU memory. VRAM budget ratio limits how much total GPU memory the planner may consume; reserve MB keeps memory free for the OS, driver, desktop, and other GPU work.");
    help_section("Models and labels",
                 "Detection needs a model file and a labels file. TensorRT uses .engine/.plan, ONNX Runtime uses .onnx, and TorchScript uses serialized TorchScript files. The input size must match the model. The precision selector is a runtime label; the model tensor type still controls FP32 or FP16 preprocessing.");
    help_section("Confidence, IoU, and input size",
                 "Confidence filters weak detections. IoU controls overlap suppression. Input size is the model resolution; the whole frame is letterboxed into that resolution, not cropped.");
    help_section("Class filtering",
                 "By default detection keeps every class the model can output. Select one or more labels in the Run Config tab to emit class IDs and restrict both CSV and annotated output.");
    help_section("Recommended presets",
                 "Use CSV Only for analytics and easiest validation. Use Safe CPU Detection when CUDA video is unavailable. Use Annotated Video for normal NVDEC/runtime/NVENC output. Use Fast GPU or Stress Test when measuring long-run hardware throughput.");
    help_section("Common errors",
                 "Check the working directory, build-win-cuda12 Release executable choice, missing vcpkg FFmpeg DLLs beside the EXE, missing TensorRT or ONNX Runtime DLLs, missing model or labels file, model/input-size mismatch, unavailable NVDEC/NVENC, locked output files, and MP4 files that are not playable until finalized.");
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
                pipelineDir / "build-win-cuda12" / "Release" / "VideoComputePipeline.exe",
                pipelineDir / "build-vs" / "Release" / "VideoComputePipeline.exe",
                pipelineDir / "build-win" / "bin" / "VideoComputePipeline.exe",
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
