/*
 * Command builder module: translates UI run configuration into a quoted
 * VideoComputePipeline command line. The UI preview, copy action, and process
 * runner all use this module so subprocess invocation stays consistent.
 */
#include "command_builder.h"

#include "path_utils.h"

#include <algorithm>
#include <sstream>

namespace vcpui {

namespace {

const std::vector<const char *> kPresetNames = {
    "People Detection - CSV Only",
    "People Detection - Staged Annotated Video",
    "Fast Staged GPU Annotated Video",
    "Safe CPU Detection",
    "NVDEC/NVENC Stress Test",
    "Custom"};

const std::vector<const char *> kTaskNames = {"filter", "detect"};
const std::vector<const char *> kDecoderNames = {"cpu", "nvdec"};
const std::vector<const char *> kEncoderNames = {"none", "h264_nvenc", "libx264", "libx264rgb", "mpeg4"};
const std::vector<const char *> kRuntimeNames = {"tensorrt", "onnxruntime", "torchscript", "auto"};
const std::vector<const char *> kBackendDeviceNames = {"cpu", "cuda"};
const std::vector<const char *> kPrecisionNames = {"fp32", "fp16"};
const std::vector<const char *> kOutputFormatNames = {"auto", "mp4", "mkv"};
const std::vector<const char *> kFilterNames = {"grayscale", "blur3x3", "blur5x5", "blur9x9", "blur13x13"};
const std::vector<const char *> kProcessModeNames = {"cpu", "gpu"};
const std::vector<const char *> kAutoIntModeNames = {"manual", "auto"};
const std::vector<const char *> kFeatureModeNames = {"auto", "on", "off"};
const std::vector<const char *> kFfmpegLogLevelNames = {"quiet", "error", "warning", "info", "debug"};

void add_arg(std::vector<std::string> &args, const char *name, const std::string &value) {
    args.emplace_back(name);
    args.emplace_back(value);
}

void add_arg(std::vector<std::string> &args, const char *name, int value) {
    args.emplace_back(name);
    args.emplace_back(std::to_string(value));
}

void add_arg(std::vector<std::string> &args, const char *name, float value) {
    std::ostringstream out;
    out.setf(std::ios::fixed);
    out.precision(3);
    out << value;
    args.emplace_back(name);
    args.emplace_back(out.str());
}

std::string join_class_ids(const std::vector<int> &ids) {
    std::vector<int> sorted = ids;
    std::sort(sorted.begin(), sorted.end());
    sorted.erase(std::unique(sorted.begin(), sorted.end()), sorted.end());

    std::ostringstream out;
    for (size_t i = 0; i < sorted.size(); ++i) {
        if (sorted[i] < 0) {
            continue;
        }
        if (out.tellp() > 0) {
            out << ',';
        }
        out << sorted[i];
    }
    return out.str();
}

std::string sanitize_family_name(std::string value) {
    std::string output;
    for (char ch : value) {
        const unsigned char c = (unsigned char)ch;
        if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || ch == '_' || ch == '-') {
            output.push_back(ch);
        } else if (ch == ' ' || ch == '.') {
            output.push_back('_');
        }
    }
    return output.empty() ? "output" : output;
}

std::string join_config_path(const std::string &folder, const std::string &file) {
    if (folder.empty()) {
        return file;
    }
    char last = folder.back();
    if (last == '\\' || last == '/') {
        return folder + file;
    }
    return folder + "/" + file;
}

const char *artifact_video_extension(const PipelineRunConfig &config) {
    if (config.outputFormat == OutputFormat::Mkv) {
        return "mkv";
    }
    if (config.outputFormat == OutputFormat::Mp4) {
        return "mp4";
    }
    return config.task == Task::Detect && config.drawBoxes ? "mkv" : "mp4";
}

void apply_output_family_paths(PipelineRunConfig &config) {
    const std::string family = sanitize_family_name(config.outputFamilyName);
    config.outputFamilyName = family;
    config.outputVideoPath = join_config_path(config.outputFolderPath, family + "_video." + artifact_video_extension(config));
    config.detectionsCsvPath = join_config_path(config.detectionsFolderPath, family + "_detections.csv");
    config.benchmarkCsvPath = join_config_path(config.benchmarkFolderPath, family + "_benchmark.csv");
}

void finalize_command(BuiltCommand &command) {
    std::ostringstream preview;
    std::wstring win32;
    for (size_t i = 0; i < command.args.size(); ++i) {
        if (i > 0) {
            preview << ' ';
            win32 += L" ";
        }
        preview << quote_arg_for_preview(command.args[i]);
        win32 += quote_arg_for_win32(utf8_to_wide(command.args[i]));
    }
    command.preview = preview.str();
    command.win32CommandLine = win32;
}

}  // namespace

const std::vector<const char *> &preset_names() { return kPresetNames; }
const std::vector<const char *> &task_names() { return kTaskNames; }
const std::vector<const char *> &decoder_names() { return kDecoderNames; }
const std::vector<const char *> &encoder_names() { return kEncoderNames; }
const std::vector<const char *> &runtime_names() { return kRuntimeNames; }
const std::vector<const char *> &backend_device_names() { return kBackendDeviceNames; }
const std::vector<const char *> &precision_names() { return kPrecisionNames; }
const std::vector<const char *> &output_format_names() { return kOutputFormatNames; }
const std::vector<const char *> &filter_names() { return kFilterNames; }
const std::vector<const char *> &process_mode_names() { return kProcessModeNames; }
const std::vector<const char *> &auto_int_mode_names() { return kAutoIntModeNames; }
const std::vector<const char *> &feature_mode_names() { return kFeatureModeNames; }
const std::vector<const char *> &ffmpeg_log_level_names() { return kFfmpegLogLevelNames; }

const char *to_cli(Task value) { return value == Task::Detect ? "detect" : "filter"; }
const char *to_cli(Decoder value) { return value == Decoder::Nvdec ? "nvdec" : "cpu"; }
const char *to_cli(Runtime value) {
    switch (value) {
        case Runtime::OnnxRuntime: return "onnxruntime";
        case Runtime::TorchScript: return "torchscript";
        case Runtime::Auto: return "auto";
        case Runtime::TensorRt:
        default: return "tensorrt";
    }
}
const char *to_cli(BackendDevice value) { return value == BackendDevice::Cuda ? "cuda" : "cpu"; }
const char *to_cli(Precision value) { return value == Precision::Fp16 ? "fp16" : "fp32"; }
const char *to_cli(OutputFormat value) {
    switch (value) {
        case OutputFormat::Mp4: return "mp4";
        case OutputFormat::Mkv: return "mkv";
        case OutputFormat::Auto:
        default: return "auto";
    }
}
const char *to_cli(Filter value) {
    switch (value) {
        case Filter::Blur3x3: return "blur3x3";
        case Filter::Blur5x5: return "blur5x5";
        case Filter::Blur9x9: return "blur9x9";
        case Filter::Blur13x13: return "blur13x13";
        case Filter::Grayscale:
        default: return "grayscale";
    }
}
const char *to_cli(ProcessMode value) { return value == ProcessMode::Gpu ? "gpu" : "cpu"; }
const char *to_cli(AutoIntMode value) { return value == AutoIntMode::Auto ? "auto" : "manual"; }
const char *to_cli(FeatureMode value) {
    switch (value) {
        case FeatureMode::On: return "on";
        case FeatureMode::Off: return "off";
        case FeatureMode::Auto:
        default: return "auto";
    }
}
const char *to_cli(Encoder value) {
    switch (value) {
        case Encoder::H264Nvenc: return "h264_nvenc";
        case Encoder::Libx264: return "libx264";
        case Encoder::Libx264Rgb: return "libx264rgb";
        case Encoder::Mpeg4: return "mpeg4";
        case Encoder::None:
        default: return "";
    }
}

void apply_preset(PipelineRunConfig &config, Preset preset) {
    config.preset = preset;
    if (preset == Preset::Custom) {
        return;
    }

    config.pipelineExePath = "../VideoComputePipeline/build-all-backends-vs/Release/VideoComputePipeline.exe";
    config.workingDirectory = "../VideoComputePipeline";
    config.inputFolderPath = "data/input";
    config.selectedInputFile = "people_4k_30min_stream_test.mp4";
    config.outputFolderPath = "data/output";
    config.detectionsFolderPath = "benchmarks/detections";
    config.benchmarkFolderPath = "benchmarks/benchmarks";
    config.outputFamilyName = "people_run";
    config.inputVideoPath = "data/input/people_4k_30min_stream_test.mp4";
    config.modelFolderPath = "models";
    config.selectedModelFile = "yolov5s.onnx";
    config.modelPath = "models/yolov5s.onnx";
    config.labelsPath = "models/coco.names";
    config.outputVideoPath = "data/output/people_run_video.mkv";
    config.detectionsCsvPath = "benchmarks/detections/people_run_detections.csv";
    config.benchmarkCsvPath = "benchmarks/benchmarks/people_run_benchmark.csv";
    config.ffmpegLogLevel = "error";
    config.task = Task::Detect;
    config.runtime = Runtime::OnnxRuntime;
    config.backendDevice = BackendDevice::Cuda;
    config.filter = Filter::Grayscale;
    config.processMode = ProcessMode::Gpu;
    config.confidence = 0.25f;
    config.iouThreshold = 0.45f;
    config.inputSize = 640;
    config.inputWidth = 640;
    config.inputHeight = 640;
    config.progressInterval = 60;
    config.boxThickness = 3;
    config.boxConfidence = 0.0f;
    config.benchmarkEnabled = true;
    config.lossless = false;
    config.decoderFallbackCpu = true;
    config.selectedClassIds.clear();
    config.autoTune = false;
    config.profileHardwareOnly = false;
    config.autoNameOutputs = true;
    config.batchSizeMode = AutoIntMode::Manual;
    config.batchSize = 1;
    config.inflightBatchesMode = AutoIntMode::Manual;
    config.inflightBatches = 1;
    config.pipelineOverlap = FeatureMode::Auto;
    config.parallelInference = FeatureMode::Auto;
    config.inferenceContextsMode = AutoIntMode::Auto;
    config.inferenceContexts = 1;
    config.targetFps = 0.0f;
    config.vramBudgetRatio = 0.375f;
    config.vramReserveMb = 0;

    switch (preset) {
        case Preset::PeopleDetectionCsvOnly:
            config.decoder = Decoder::Cpu;
            config.encoder = Encoder::None;
            config.runtime = Runtime::OnnxRuntime;
            config.backendDevice = BackendDevice::Cuda;
            config.selectedModelFile = "yolov5s.onnx";
            config.modelPath = "models/yolov5s.onnx";
            config.outputFamilyName = "people_csv";
            config.precision = Precision::Fp32;
            config.outputFormat = OutputFormat::Auto;
            config.drawBoxes = false;
            config.maxFrames = 0;
            break;
        case Preset::PeopleDetectionAnnotatedVideo:
            config.decoder = Decoder::Nvdec;
            config.encoder = Encoder::H264Nvenc;
            config.runtime = Runtime::OnnxRuntime;
            config.backendDevice = BackendDevice::Cuda;
            config.selectedModelFile = "yolov5s.onnx";
            config.modelPath = "models/yolov5s.onnx";
            config.outputFamilyName = "people_annotated";
            config.precision = Precision::Fp32;
            config.outputFormat = OutputFormat::Mkv;
            config.drawBoxes = true;
            config.maxFrames = 0;
            config.autoTune = true;
            config.batchSizeMode = AutoIntMode::Auto;
            config.inflightBatchesMode = AutoIntMode::Auto;
            break;
        case Preset::FastGpuAnnotatedVideo:
            config.decoder = Decoder::Nvdec;
            config.encoder = Encoder::H264Nvenc;
            config.runtime = Runtime::TensorRt;
            config.backendDevice = BackendDevice::Cuda;
            config.selectedModelFile = "yolov5s_trt11.engine";
            config.modelPath = "models/yolov5s_trt11.engine";
            config.outputFamilyName = "fast_gpu_annotated";
            config.precision = Precision::Fp16;
            config.outputFormat = OutputFormat::Mkv;
            config.drawBoxes = true;
            config.maxFrames = 0;
            config.autoTune = true;
            config.batchSizeMode = AutoIntMode::Manual;
            config.batchSize = 2;
            config.inflightBatchesMode = AutoIntMode::Manual;
            config.inflightBatches = 3;
            config.pipelineOverlap = FeatureMode::On;
            config.parallelInference = FeatureMode::On;
            config.inferenceContextsMode = AutoIntMode::Manual;
            config.inferenceContexts = 2;
            config.targetFps = 60.0f;
            break;
        case Preset::SafeCpuDetection:
            config.decoder = Decoder::Cpu;
            config.encoder = Encoder::None;
            config.runtime = Runtime::OnnxRuntime;
            config.backendDevice = BackendDevice::Cpu;
            config.selectedModelFile = "yolov5s.onnx";
            config.modelPath = "models/yolov5s.onnx";
            config.outputFamilyName = "safe_cpu_detection";
            config.precision = Precision::Fp32;
            config.outputFormat = OutputFormat::Auto;
            config.drawBoxes = false;
            config.maxFrames = 120;
            break;
        case Preset::NvdecNvencStressTest:
            config.decoder = Decoder::Nvdec;
            config.decoderFallbackCpu = false;
            config.encoder = Encoder::H264Nvenc;
            config.runtime = Runtime::TensorRt;
            config.backendDevice = BackendDevice::Cuda;
            config.selectedModelFile = "yolov5s_trt11.engine";
            config.modelPath = "models/yolov5s_trt11.engine";
            config.outputFamilyName = "nvdec_nvenc_stress";
            config.precision = Precision::Fp32;
            config.outputFormat = OutputFormat::Mkv;
            config.drawBoxes = true;
            config.maxFrames = 0;
            config.autoTune = true;
            config.batchSizeMode = AutoIntMode::Manual;
            config.batchSize = 2;
            config.inflightBatchesMode = AutoIntMode::Manual;
            config.inflightBatches = 3;
            config.pipelineOverlap = FeatureMode::On;
            config.parallelInference = FeatureMode::On;
            config.inferenceContextsMode = AutoIntMode::Manual;
            config.inferenceContexts = 2;
            config.targetFps = 60.0f;
            break;
        case Preset::Custom:
        default:
            break;
    }
    if (config.autoNameOutputs) {
        apply_output_family_paths(config);
    }
}

std::vector<std::string> validate_config(const PipelineRunConfig &config) {
    std::vector<std::string> issues;
    if (!file_exists(config.pipelineExePath)) {
        issues.emplace_back("Pipeline executable was not found.");
    }
    if (!directory_exists(config.workingDirectory)) {
        issues.emplace_back("Working directory was not found.");
    }
    if (config.task == Task::Detect && config.profileHardwareOnly) {
        return issues;
    }
    if (!directory_exists(resolve_path(config.workingDirectory, config.inputFolderPath))) {
        issues.emplace_back("Input folder was not found relative to the pipeline working directory.");
    }
    if (!config.outputFolderPath.empty() &&
        !directory_exists(resolve_path(config.workingDirectory, config.outputFolderPath))) {
        issues.emplace_back("Output folder was not found relative to the pipeline working directory.");
    }
    if (!config.detectionsFolderPath.empty() &&
        !directory_exists(resolve_path(config.workingDirectory, config.detectionsFolderPath))) {
        issues.emplace_back("Detections folder was not found relative to the pipeline working directory.");
    }
    if (!config.benchmarkFolderPath.empty() &&
        !directory_exists(resolve_path(config.workingDirectory, config.benchmarkFolderPath))) {
        issues.emplace_back("Benchmark folder was not found relative to the pipeline working directory.");
    }
    if (!file_exists(resolve_path(config.workingDirectory, config.inputVideoPath))) {
        issues.emplace_back("Input video was not found relative to the pipeline working directory.");
    }
    if (config.task == Task::Detect) {
        if (config.runtime == Runtime::TensorRt && config.backendDevice == BackendDevice::Cpu) {
            issues.emplace_back("TensorRT is CUDA-only. Select inference device cuda or choose ONNX Runtime/TorchScript for CPU inference.");
        }
        if (config.inputWidth <= 0 || config.inputHeight <= 0) {
            issues.emplace_back("Input width and input height must be greater than 0.");
        }
        if (config.drawBoxes &&
            !(config.decoder == Decoder::Nvdec &&
              config.backendDevice == BackendDevice::Cuda &&
              config.encoder == Encoder::H264Nvenc)) {
            issues.emplace_back("Annotated detection currently requires decoder nvdec, inference device cuda, and encoder h264_nvenc. Disable Draw boxes for CSV-only detection.");
        }
        if (!file_exists(resolve_path(config.workingDirectory, config.modelPath))) {
            issues.emplace_back("Model file was not found relative to the pipeline working directory.");
        }
        if (!directory_exists(resolve_path(config.workingDirectory, config.modelFolderPath))) {
            issues.emplace_back("Model folder was not found relative to the pipeline working directory.");
        }
        if (!file_exists(resolve_path(config.workingDirectory, config.labelsPath))) {
            issues.emplace_back("Labels file was not found relative to the pipeline working directory.");
        }
        if (config.benchmarkEnabled && config.detectionsCsvPath == config.benchmarkCsvPath) {
            issues.emplace_back("Detections CSV and benchmark CSV should use different paths.");
        }
        const std::string exeDir = parent_directory(config.pipelineExePath);
        const char *ffmpegDlls[] = {
            "avcodec-62.dll",
            "avformat-62.dll",
            "avutil-60.dll",
            "swscale-9.dll",
            "swresample-6.dll",
            "avdevice-62.dll",
            "avfilter-11.dll",
        };
        for (const char *dll : ffmpegDlls) {
            if (!file_exists(resolve_path(exeDir, dll))) {
                issues.emplace_back(std::string("Expected vcpkg FFmpeg DLL beside the pipeline executable is missing: ") + dll);
            }
        }
    }
    return issues;
}

std::string quote_arg_for_preview(const std::string &arg) {
    if (arg.empty()) {
        return "\"\"";
    }
    const bool needsQuote = arg.find_first_of(" \t\"") != std::string::npos;
    if (!needsQuote) {
        return arg;
    }
    std::string out = "\"";
    for (char ch : arg) {
        if (ch == '"') {
            out += "\\\"";
        } else {
            out += ch;
        }
    }
    out += '"';
    return out;
}

std::wstring quote_arg_for_win32(const std::wstring &arg) {
    if (arg.empty()) {
        return L"\"\"";
    }
    bool needsQuote = false;
    for (wchar_t ch : arg) {
        if (ch == L' ' || ch == L'\t' || ch == L'"') {
            needsQuote = true;
            break;
        }
    }
    if (!needsQuote) {
        return arg;
    }

    std::wstring result = L"\"";
    unsigned int backslashes = 0;
    for (wchar_t ch : arg) {
        if (ch == L'\\') {
            ++backslashes;
        } else if (ch == L'"') {
            result.append(backslashes * 2u + 1u, L'\\');
            result += ch;
            backslashes = 0;
        } else {
            result.append(backslashes, L'\\');
            backslashes = 0;
            result += ch;
        }
    }
    result.append(backslashes * 2u, L'\\');
    result += L'"';
    return result;
}

BuiltCommand build_command(const PipelineRunConfig &inputConfig) {
    PipelineRunConfig config = inputConfig;
    if (config.autoNameOutputs) {
        apply_output_family_paths(config);
    }

    BuiltCommand command;
    command.args.emplace_back(config.pipelineExePath);
    add_arg(command.args, "--task", to_cli(config.task));

    if (config.task == Task::Detect && config.profileHardwareOnly) {
        command.args.emplace_back("--profile-hardware");
        add_arg(command.args, "--ffmpeg-log-level", config.ffmpegLogLevel);
        finalize_command(command);
        return command;
    }

    add_arg(command.args, "--input", config.inputVideoPath);
    if (config.benchmarkEnabled) {
        add_arg(command.args, "--benchmark", config.benchmarkCsvPath);
    } else {
        command.args.emplace_back("--no-benchmark");
    }

    if (config.task == Task::Detect) {
        add_arg(command.args, "--decoder", to_cli(config.decoder));
        if (!config.decoderFallbackCpu) {
            command.args.emplace_back("--decoder-fallback");
            command.args.emplace_back("none");
        }
        add_arg(command.args, "--model", config.modelPath);
        add_arg(command.args, "--labels", config.labelsPath);
        add_arg(command.args, "--detections", config.detectionsCsvPath);
        add_arg(command.args, "--confidence", config.confidence);
        add_arg(command.args, "--iou-threshold", config.iouThreshold);
        if (config.inputWidth == config.inputHeight) {
            add_arg(command.args, "--input-size", config.inputWidth);
        } else {
            add_arg(command.args, "--input-width", config.inputWidth);
            add_arg(command.args, "--input-height", config.inputHeight);
        }
        add_arg(command.args, "--runtime", to_cli(config.runtime));
        add_arg(command.args, "--backend-device", to_cli(config.backendDevice));
        add_arg(command.args, "--precision", to_cli(config.precision));
        if (config.autoTune) {
            command.args.emplace_back("--auto-tune");
        }
        if (config.batchSizeMode == AutoIntMode::Auto) {
            add_arg(command.args, "--batch-size", "auto");
        } else {
            add_arg(command.args, "--batch-size", config.batchSize);
        }
        if (config.inflightBatchesMode == AutoIntMode::Auto) {
            add_arg(command.args, "--inflight-batches", "auto");
        } else {
            add_arg(command.args, "--inflight-batches", config.inflightBatches);
        }
        if (config.targetFps > 0.0f) {
            add_arg(command.args, "--target-fps", config.targetFps);
        }
        if (config.vramBudgetRatio != 0.375f) {
            add_arg(command.args, "--vram-budget-ratio", config.vramBudgetRatio);
        }
        if (config.vramReserveMb > 0) {
            add_arg(command.args, "--vram-reserve-mb", config.vramReserveMb);
        }
        add_arg(command.args, "--pipeline-overlap", to_cli(config.pipelineOverlap));
        add_arg(command.args, "--parallel-inference", to_cli(config.parallelInference));
        if (config.inferenceContextsMode == AutoIntMode::Auto) {
            add_arg(command.args, "--inference-contexts", "auto");
        } else {
            add_arg(command.args, "--inference-contexts", config.inferenceContexts);
        }
        const std::string classIds = join_class_ids(config.selectedClassIds);
        if (!classIds.empty()) {
            add_arg(command.args, "--class-ids", classIds);
        }

        if (config.drawBoxes) {
            command.args.emplace_back("--draw-boxes");
            add_arg(command.args, "--output", config.outputVideoPath);
            add_arg(command.args, "--output-format", to_cli(config.outputFormat));
            if (config.encoder != Encoder::None) {
                add_arg(command.args, "--encoder", to_cli(config.encoder));
            }
            add_arg(command.args, "--box-thickness", config.boxThickness);
            add_arg(command.args, "--box-confidence", config.boxConfidence);
        }
    } else {
        add_arg(command.args, "--mode", to_cli(config.processMode));
        add_arg(command.args, "--filter", to_cli(config.filter));
        add_arg(command.args, "--output", config.outputVideoPath);
        add_arg(command.args, "--output-format", to_cli(config.outputFormat));
        if (config.encoder != Encoder::None) {
            add_arg(command.args, "--encoder", to_cli(config.encoder));
        }
        if (config.lossless) {
            command.args.emplace_back("--lossless");
        }
    }

    if (config.maxFrames > 0) {
        add_arg(command.args, "--max-frames", config.maxFrames);
    }
    add_arg(command.args, "--ffmpeg-log-level", config.ffmpegLogLevel);
    add_arg(command.args, "--progress-interval", config.progressInterval);

    finalize_command(command);
    return command;
}

}  // namespace vcpui
