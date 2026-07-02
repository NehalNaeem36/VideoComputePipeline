#ifndef VCP_UI_PIPELINE_RUN_CONFIG_H
#define VCP_UI_PIPELINE_RUN_CONFIG_H

#include <string>
#include <vector>

namespace vcpui {

enum class Preset {
    PeopleDetectionCsvOnly = 0,
    PeopleDetectionAnnotatedVideo,
    FastGpuAnnotatedVideo,
    SafeCpuDetection,
    NvdecNvencStressTest,
    Custom
};

enum class Task { Filter = 0, Detect };
enum class Decoder { Cpu = 0, Nvdec };
enum class Encoder { None = 0, H264Nvenc, Libx264, Libx264Rgb, Mpeg4 };
enum class Runtime { TensorRt = 0, OnnxRuntime, TorchScript, Auto };
enum class Precision { Fp32 = 0, Fp16 };
enum class OutputFormat { Auto = 0, Mp4, Mkv };
enum class Filter { Grayscale = 0, Blur3x3, Blur5x5, Blur9x9, Blur13x13 };
enum class ProcessMode { Cpu = 0, Gpu };
enum class AutoIntMode { Manual = 0, Auto };
enum class FeatureMode { Auto = 0, On, Off };

struct PipelineRunConfig {
    std::string pipelineExePath = "..\\VideoComputePipeline\\build-win-cuda12\\Release\\VideoComputePipeline.exe";
    std::string workingDirectory = "..\\VideoComputePipeline";
    std::string inputVideoPath = "data\\input\\people_4k_30min_stream_test.mp4";
    std::string modelPath = "models\\yolov5s_trt11.engine";
    std::string labelsPath = "models\\coco.names";
    std::string outputVideoPath = "data\\output\\people_annotated_live.mkv";
    std::string detectionsCsvPath = "benchmarks\\people_detections.csv";
    std::string benchmarkCsvPath = "benchmarks\\people_full_inference.csv";
    std::string ffmpegLogLevel = "error";
    std::vector<int> selectedClassIds;

    Preset preset = Preset::PeopleDetectionAnnotatedVideo;
    Task task = Task::Detect;
    Decoder decoder = Decoder::Nvdec;
    Encoder encoder = Encoder::H264Nvenc;
    Runtime runtime = Runtime::OnnxRuntime;
    Precision precision = Precision::Fp32;
    OutputFormat outputFormat = OutputFormat::Mkv;
    Filter filter = Filter::Grayscale;
    ProcessMode processMode = ProcessMode::Gpu;
    AutoIntMode batchSizeMode = AutoIntMode::Manual;
    AutoIntMode inflightBatchesMode = AutoIntMode::Manual;
    AutoIntMode inferenceContextsMode = AutoIntMode::Auto;
    FeatureMode pipelineOverlap = FeatureMode::Auto;
    FeatureMode parallelInference = FeatureMode::Auto;

    bool decoderFallbackCpu = true;
    bool drawBoxes = true;
    bool benchmarkEnabled = true;
    bool lossless = false;
    bool autoTune = false;
    bool profileHardwareOnly = false;

    int boxThickness = 3;
    int batchSize = 1;
    int inflightBatches = 1;
    int inferenceContexts = 1;
    int vramReserveMb = 0;
    float boxConfidence = 0.0f;
    float confidence = 0.25f;
    float iouThreshold = 0.45f;
    float targetFps = 0.0f;
    float vramBudgetRatio = 0.375f;
    int inputSize = 640;
    int maxFrames = 0;
    int progressInterval = 60;
};

const std::vector<const char *> &preset_names();
const std::vector<const char *> &task_names();
const std::vector<const char *> &decoder_names();
const std::vector<const char *> &encoder_names();
const std::vector<const char *> &runtime_names();
const std::vector<const char *> &precision_names();
const std::vector<const char *> &output_format_names();
const std::vector<const char *> &filter_names();
const std::vector<const char *> &process_mode_names();
const std::vector<const char *> &auto_int_mode_names();
const std::vector<const char *> &feature_mode_names();
const std::vector<const char *> &ffmpeg_log_level_names();

void apply_preset(PipelineRunConfig &config, Preset preset);
std::vector<std::string> validate_config(const PipelineRunConfig &config);

const char *to_cli(Task value);
const char *to_cli(Decoder value);
const char *to_cli(Encoder value);
const char *to_cli(Runtime value);
const char *to_cli(Precision value);
const char *to_cli(OutputFormat value);
const char *to_cli(Filter value);
const char *to_cli(ProcessMode value);
const char *to_cli(AutoIntMode value);
const char *to_cli(FeatureMode value);

}  // namespace vcpui

#endif
