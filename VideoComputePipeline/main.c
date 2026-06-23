#include "config.h"
#include "benchmark/matrix_report.h"
#include "inference/backend_registry.h"
#include "pipeline/pipeline_config.h"
#include "pipeline/pipeline_runner.h"
#include "utils/logger.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void print_common_options(void) {
    printf("Common options:\n");
    printf("  --input path                 Input MP4 path\n");
    printf("  --benchmark path             Benchmark CSV path\n");
    printf("  --max-frames N               Stop after N frames, 0 means no limit\n");
    printf("  --decoder-threads N          FFmpeg decoder worker threads\n");
    printf("  --progress-interval N        Log progress every N completed frames, 0 disables\n");
    printf("  --no-progress                Disable progress logging\n");
    printf("  --ffmpeg-log-level level     quiet, error, warning, info, or debug\n");
    printf("  --output-format auto|mp4|mkv Force output container extension\n");
    printf("  --no-benchmark               Disable benchmark output\n");
    printf("  --version                    Show version information\n");
    printf("  --list-backends              Show compiled and available inference runtimes\n");
    printf("  --model-info                 Show selected model/runtime information and exit\n");
}

static void print_usage(const char *program_name) {
    printf("Usage: %s --task filter [filter options]\n", program_name);
    printf("       %s --task detect [detection options]\n", program_name);
    printf("\nTasks:\n");
    printf("  filter                       Decode, process, and encode a video\n");
    printf("  detect                       Decode NV12 frames and stream YOLO detections\n");
    printf("\n");
    print_common_options /* module: app/main */ ();
    printf("\nTask help:\n");
    printf("  --help-filter                Show filter pipeline options\n");
    printf("  --help-detect                Show detection pipeline options\n");
    printf("  --help-matrix                Show CPU/GPU matrix report usage\n");
    printf("\nExamples:\n");
    printf("  %s --task filter --mode gpu --filter blur5x5 --input data/input/sample.mp4 --output data/output/blur.mp4\n", program_name);
    printf("  %s --task detect --input data/input/sample.mp4 --model models/yolov5s_trt11.engine --labels models/coco.names\n", program_name);
}

static void print_filter_usage(const char *program_name) {
    printf("Usage: %s --task filter [options]\n\n", program_name);
    print_common_options /* module: app/main */ ();
    printf("\nFilter options:\n");
    printf("  --output path                Output video path\n");
    printf("  --mode cpu|gpu               Processing mode\n");
    printf("  --filter name                grayscale, blur3x3, blur5x5, blur9x9, blur13x13\n");
    printf("  --encoder name               libx264, libx264rgb, h264_nvenc, or mpeg4\n");
    printf("  --encoder-threads N          FFmpeg encoder worker threads\n");
    printf("  --processor-workers N        CPU processor workers, GPU mode uses one worker\n");
    printf("  --frame-slots N              Frames buffered between pipeline stages\n");
    printf("  --lossless                   Use lossless encoder settings when supported\n");
    printf("  --lossy                      Use lossy encoder settings\n");
    printf("  --memory-profile name        auto, low, balanced, or manual\n");
    printf("  --memory-budget-mb N         Optional CPU frame-pool budget, 0 means automatic\n");
}

static void print_detect_usage(const char *program_name) {
    printf("Usage: %s --task detect [options]\n\n", program_name);
    print_common_options /* module: app/main */ ();
    printf("\nDetection options:\n");
    printf("  --decoder cpu|nvdec          CPU fallback decode or GPU-resident NVDEC path\n");
    printf("  --decoder-fallback cpu|none  Fallback when NVDEC is unavailable\n");
    printf("  --no-decoder-fallback        Fail instead of falling back to CPU decode\n");
    printf("  --model path                 TensorRT engine path\n");
    printf("  --labels path                Class label file\n");
    printf("  --detections path            Detection CSV output path\n");
    printf("  --draw-boxes                 Draw detection boxes and write annotated video\n");
    printf("  --no-draw-boxes              Disable annotated video output\n");
    printf("  --output path                Annotated video output path when --draw-boxes is set\n");
    printf("  --encoder name               h264_nvenc for GPU-resident annotated output\n");
    printf("  --box-thickness N            Bounding box border thickness\n");
    printf("  --box-confidence value       Minimum confidence for drawing boxes\n");
    printf("  --confidence value           Detection confidence threshold, 0.0 to 1.0\n");
    printf("  --iou-threshold value        Detection NMS IoU threshold, 0.0 to 1.0\n");
    printf("  --classes names              Comma-separated label names to keep, for example person,car\n");
    printf("  --class-ids ids              Comma-separated numeric class IDs to keep, for example 0,2\n");
    printf("  --input-size N               Detector input size, default 640\n");
    printf("  --runtime name               auto, tensorrt, onnxruntime, or torchscript\n");
    printf("  --backend-device name        cuda or cpu, default cuda\n");
    printf("  --allow-host-backend         Allow explicit slow host backend fallback\n");
    printf("  --model-info                 Print model/runtime information and exit\n");
    printf("  --model-type name            auto or yolov5\n");
    printf("  --inference-backend name     Backward-compatible alias for --runtime\n");
    printf("  --precision name             fp16 or fp32 runtime label\n");
    printf("  --batch-size auto|N          Frames per detection batch, default 1\n");
    printf("  --inflight-batches auto|N    Batches allowed in flight, default 1\n");
    printf("  --auto-tune                  Select batch settings from video/GPU/model profile\n");
    printf("  --target-fps N               Target FPS for transfer budget calculations\n");
    printf("  --vram-budget-ratio R        Fraction of total VRAM usable by pipeline, default 0.375\n");
    printf("  --vram-reserve-mb N          Explicit VRAM reserve, 0 means automatic reserve\n");
    printf("  --profile-hardware           Print CUDA/VRAM/bandwidth profile and exit\n");
    printf("  --pipeline-overlap auto|on|off Overlap detection stages when supported\n");
    printf("  --parallel-inference auto|on|off Use multiple TensorRT contexts when supported\n");
    printf("  --inference-contexts auto|N  TensorRT execution contexts for parallel inference\n");
}

static void print_matrix_usage(const char *program_name) {
    printf("Usage: %s --matrix-report cpu.csv gpu.csv\n\n", program_name);
    printf("Reads two benchmark CSV files and prints a CPU vs GPU summary.\n");
}

static void print_version(void) {
    printf("%s %s\n", PROJECT_NAME, PROJECT_VERSION);
}

int main(int argc, char **argv) {

    // cli arguments are passed to the pipeline config parser, but we also handle some special cases here
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--help") == 0) {
            print_usage /* module: app/main */ (argv[0]);
            return EXIT_SUCCESS;
        }

        if (strcmp(argv[i], "--help-filter") == 0) {
            print_filter_usage /* module: app/main */ (argv[0]);
            return EXIT_SUCCESS;
        }

        if (strcmp(argv[i], "--help-detect") == 0) {
            print_detect_usage /* module: app/main */ (argv[0]);
            return EXIT_SUCCESS;
        }

        if (strcmp(argv[i], "--help-matrix") == 0) {
            print_matrix_usage /* module: app/main */ (argv[0]);
            return EXIT_SUCCESS;
        }

        if (strcmp(argv[i], "--version") == 0) {
            print_version /* module: app/main */ ();
            return EXIT_SUCCESS;
        }

        if (strcmp(argv[i], "--matrix-report") == 0) {
            if (i + 2 >= argc) {
                log_error /* module: utils/logger */ ("--matrix-report requires CPU and GPU benchmark CSV paths");
                print_matrix_usage /* module: app/main */ (argv[0]);
                return EXIT_FAILURE;
            }
            MatrixReportStats cpu;
            MatrixReportStats gpu;
            if (matrix_report_read_csv_summary /* module: benchmark/matrix_report */ (argv[i + 1], &cpu) != 0 ||
                matrix_report_read_csv_summary /* module: benchmark/matrix_report */ (argv[i + 2], &gpu) != 0) {
                log_error /* module: utils/logger */ ("failed to read matrix report inputs");
                return EXIT_FAILURE;
            }
            matrix_report_print_comparison /* module: benchmark/matrix_report */ (&cpu, &gpu);
            return EXIT_SUCCESS;
        }
    }

    PipelineConfig config;
    //config store deffault at first
    pipeline_config_default /* module: pipeline/pipeline_config */ (&config);


    //then we parse the cli arguments and override the config with them
    if (pipeline_config_parse_args /* module: pipeline/pipeline_config */ (&config, argc, argv) != 0) {
        /*here pipeline_config_parse_args tried to parse given args and build the config but failed */
        log_error /* module: utils/logger */ ("%s", pipeline_config_last_error /* module: pipeline/pipeline_config */ ());
        printf("Run %s --help for usage.\n", argv[0]);
        return EXIT_FAILURE;
    }

    if (config.list_backends) {
        inference_backend_registry_print /* module: inference/backend_registry */ (stdout);
        return EXIT_SUCCESS;
    }

    if (config.model_info) {
        inference_backend_model_info_print /* module: inference/backend_registry */ (stdout,
                                                                                    config.model_path,
                                                                                    config.labels_path,
                                                                                    config.runtime,
                                                                                    config.backend_device,
                                                                                    config.model_type);
        return EXIT_SUCCESS;
    }
    /* if config created sucessfully 
            print config 
            run pipeline  and logg error in result*/
    printf("%s %s\n", PROJECT_NAME, PROJECT_VERSION);
    printf("configuration:\n");
    pipeline_config_print /* module: pipeline/pipeline_config */ (&config);
    fflush(stdout);

    log_info /* module: utils/logger */ ("starting pipeline");

    /*  pipeline_run is the main execution entry point*/
    const int result = pipeline_run /* module: pipeline/pipeline_runner */ (&config);
    if (result != 0) {
        log_error /* module: utils/logger */ ("pipeline failed with code %d", result);
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
