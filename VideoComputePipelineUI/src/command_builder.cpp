#include "command_builder.h"

#include "path_utils.h"

#include <algorithm>
#include <sstream>

namespace vcpui {

namespace {

const std::vector<const char *> kPresetNames = {
    "People Detection - CSV Only",
    "People Detection - Annotated Video",
    "Fast GPU Annotated Video",
    "Safe CPU Detection",
    "NVDEC/NVENC Stress Test",
    "Custom"};

const std::vector<const char *> kTaskNames = {"filter", "detect"};
const std::vector<const char *> kDecoderNames = {"cpu", "nvdec"};
const std::vector<const char *> kEncoderNames = {"none", "h264_nvenc", "libx264", "libx264rgb", "mpeg4"};
const std::vector<const char *> kPrecisionNames = {"fp32", "fp16"};
const std::vector<const char *> kOutputFormatNames = {"auto", "mp4", "mkv"};
const std::vector<const char *> kFilterNames = {"grayscale", "blur3x3", "blur5x5", "blur9x9", "blur13x13"};
const std::vector<const char *> kProcessModeNames = {"cpu", "gpu"};
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

}  // namespace

const std::vector<const char *> &preset_names() { return kPresetNames; }
const std::vector<const char *> &task_names() { return kTaskNames; }
const std::vector<const char *> &decoder_names() { return kDecoderNames; }
const std::vector<const char *> &encoder_names() { return kEncoderNames; }
const std::vector<const char *> &precision_names() { return kPrecisionNames; }
const std::vector<const char *> &output_format_names() { return kOutputFormatNames; }
const std::vector<const char *> &filter_names() { return kFilterNames; }
const std::vector<const char *> &process_mode_names() { return kProcessModeNames; }
const std::vector<const char *> &ffmpeg_log_level_names() { return kFfmpegLogLevelNames; }

const char *to_cli(Task value) { return value == Task::Detect ? "detect" : "filter"; }
const char *to_cli(Decoder value) { return value == Decoder::Nvdec ? "nvdec" : "cpu"; }
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

    config.pipelineExePath = "..\\VideoComputePipeline\\build-win\\bin\\VideoComputePipeline.exe";
    config.workingDirectory = "..\\VideoComputePipeline";
    config.inputVideoPath = "data\\input\\people_4k_30min_stream_test.mp4";
    config.modelPath = "models\\yolov5s_trt11.engine";
    config.labelsPath = "models\\coco.names";
    config.outputVideoPath = "data\\output\\people_annotated_live.mkv";
    config.detectionsCsvPath = "benchmarks\\people_detections.csv";
    config.benchmarkCsvPath = "benchmarks\\people_full_inference.csv";
    config.ffmpegLogLevel = "error";
    config.task = Task::Detect;
    config.filter = Filter::Grayscale;
    config.processMode = ProcessMode::Gpu;
    config.confidence = 0.25f;
    config.iouThreshold = 0.45f;
    config.inputSize = 640;
    config.progressInterval = 60;
    config.boxThickness = 3;
    config.boxConfidence = 0.0f;
    config.benchmarkEnabled = true;
    config.lossless = false;
    config.decoderFallbackCpu = true;
    config.selectedClassIds.clear();

    switch (preset) {
        case Preset::PeopleDetectionCsvOnly:
            config.decoder = Decoder::Cpu;
            config.encoder = Encoder::None;
            config.precision = Precision::Fp32;
            config.outputFormat = OutputFormat::Auto;
            config.drawBoxes = false;
            config.maxFrames = 0;
            break;
        case Preset::PeopleDetectionAnnotatedVideo:
            config.decoder = Decoder::Nvdec;
            config.encoder = Encoder::H264Nvenc;
            config.precision = Precision::Fp32;
            config.outputFormat = OutputFormat::Mkv;
            config.drawBoxes = true;
            config.maxFrames = 0;
            break;
        case Preset::FastGpuAnnotatedVideo:
            config.decoder = Decoder::Nvdec;
            config.encoder = Encoder::H264Nvenc;
            config.precision = Precision::Fp16;
            config.outputFormat = OutputFormat::Mkv;
            config.drawBoxes = true;
            config.maxFrames = 0;
            break;
        case Preset::SafeCpuDetection:
            config.decoder = Decoder::Cpu;
            config.encoder = Encoder::None;
            config.precision = Precision::Fp32;
            config.outputFormat = OutputFormat::Auto;
            config.drawBoxes = false;
            config.maxFrames = 120;
            break;
        case Preset::NvdecNvencStressTest:
            config.decoder = Decoder::Nvdec;
            config.decoderFallbackCpu = false;
            config.encoder = Encoder::H264Nvenc;
            config.precision = Precision::Fp32;
            config.outputFormat = OutputFormat::Mkv;
            config.drawBoxes = true;
            config.maxFrames = 0;
            break;
        case Preset::Custom:
        default:
            break;
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
    if (!file_exists(resolve_path(config.workingDirectory, config.inputVideoPath))) {
        issues.emplace_back("Input video was not found relative to the pipeline working directory.");
    }
    if (config.task == Task::Detect) {
        if (!file_exists(resolve_path(config.workingDirectory, config.modelPath))) {
            issues.emplace_back("TensorRT model engine was not found relative to the pipeline working directory.");
        }
        if (!file_exists(resolve_path(config.workingDirectory, config.labelsPath))) {
            issues.emplace_back("Labels file was not found relative to the pipeline working directory.");
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

BuiltCommand build_command(const PipelineRunConfig &config) {
    BuiltCommand command;
    command.args.emplace_back(config.pipelineExePath);
    add_arg(command.args, "--task", to_cli(config.task));
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
        add_arg(command.args, "--input-size", config.inputSize);
        add_arg(command.args, "--inference-backend", "tensorrt");
        add_arg(command.args, "--precision", to_cli(config.precision));
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
    return command;
}

}  // namespace vcpui
