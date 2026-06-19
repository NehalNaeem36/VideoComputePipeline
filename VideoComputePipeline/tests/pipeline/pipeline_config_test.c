#include "pipeline/pipeline_config.h"

#include <stdio.h>
#include <string.h>

#define TEST_ASSERT(expr) \
    do { \
        if (!(expr)) { \
            fprintf(stderr, "Assertion failed: %s:%d: %s\n", __FILE__, __LINE__, #expr); \
            return 1; \
        } \
    } while (0)

static int pipeline_config_test_defaults(void) {
    PipelineConfig config;
    pipeline_config_default /* module: pipeline/pipeline_config */ (&config);

    TEST_ASSERT(strcmp(config.input_path, DEFAULT_INPUT_PATH) == 0);
    TEST_ASSERT(strcmp(config.output_path, DEFAULT_OUTPUT_PATH) == 0);
    TEST_ASSERT(strcmp(config.benchmark_path, DEFAULT_BENCHMARK_PATH) == 0);
    TEST_ASSERT(strcmp(config.detections_path, DEFAULT_DETECTIONS_PATH) == 0);
    TEST_ASSERT(strcmp(config.model_path, DEFAULT_MODEL_PATH) == 0);
    TEST_ASSERT(strcmp(config.labels_path, DEFAULT_LABELS_PATH) == 0);
    TEST_ASSERT(strcmp(config.encoder_name, DEFAULT_ENCODER_NAME) == 0);
    TEST_ASSERT(config.task == PIPELINE_TASK_FILTER);
    TEST_ASSERT(config.mode == PROCESS_CPU);
    TEST_ASSERT(config.filter == FILTER_GRAYSCALE);
    TEST_ASSERT(config.max_frames == DEFAULT_MAX_FRAMES);
    TEST_ASSERT(config.enable_benchmark == DEFAULT_ENABLE_BENCHMARK);
    TEST_ASSERT(config.lossless_output == DEFAULT_LOSSLESS_OUTPUT);
    TEST_ASSERT(config.memory_profile == MEMORY_PROFILE_AUTO);
    TEST_ASSERT(config.memory_budget_mb == DEFAULT_MEMORY_BUDGET_MB);
    TEST_ASSERT(config.frame_slots == DEFAULT_FRAME_SLOTS);
    TEST_ASSERT(config.decoder_threads == DEFAULT_DECODER_THREADS);
    TEST_ASSERT(config.encoder_threads == DEFAULT_ENCODER_THREADS);
    TEST_ASSERT(config.processor_workers == DEFAULT_PROCESSOR_WORKERS);
    TEST_ASSERT(config.confidence_threshold == DEFAULT_DETECTION_CONFIDENCE);
    TEST_ASSERT(config.iou_threshold == DEFAULT_DETECTION_IOU_THRESHOLD);
    TEST_ASSERT(config.inference_input_size == DEFAULT_INFERENCE_INPUT_SIZE);
    TEST_ASSERT(config.progress_interval == DEFAULT_PROGRESS_INTERVAL);
    TEST_ASSERT(strcmp(config.ffmpeg_log_level, DEFAULT_FFMPEG_LOG_LEVEL) == 0);
    return 0;
}

static int pipeline_config_test_parse_args(void) {
    char *argv[] = {
        "VideoComputePipeline",
        "--input", "input.mp4",
        "--output", "output.mp4",
        "--benchmark", "bench.csv",
        "--task", "detect",
        "--model", "models/custom.engine",
        "--labels", "models/custom.names",
        "--detections", "benchmarks/detections.csv",
        "--confidence", "0.35",
        "--iou-threshold", "0.50",
        "--input-size", "512",
        "--inference-backend", "tensorrt",
        "--precision", "fp16",
        "--encoder", "h264_nvenc",
        "--lossless",
        "--mode", "gpu",
        "--filter", "blur13x13",
        "--max-frames", "12",
        "--frame-slots", "8",
        "--decoder-threads", "3",
        "--encoder-threads", "5",
        "--processor-workers", "6",
        "--memory-profile", "manual",
        "--memory-budget-mb", "512",
        "--progress-interval", "42",
        "--ffmpeg-log-level", "error",
        "--no-benchmark"
    };
    const int argc = (int)(sizeof(argv) / sizeof(argv[0]));

    PipelineConfig config;
    pipeline_config_default /* module: pipeline/pipeline_config */ (&config);

    TEST_ASSERT(pipeline_config_parse_args /* module: pipeline/pipeline_config */ (&config, argc, argv) == 0);
    TEST_ASSERT(strcmp(config.input_path, "input.mp4") == 0);
    TEST_ASSERT(strcmp(config.output_path, "output.mp4") == 0);
    TEST_ASSERT(strcmp(config.benchmark_path, "bench.csv") == 0);
    TEST_ASSERT(strcmp(config.detections_path, "benchmarks/detections.csv") == 0);
    TEST_ASSERT(strcmp(config.model_path, "models/custom.engine") == 0);
    TEST_ASSERT(strcmp(config.labels_path, "models/custom.names") == 0);
    TEST_ASSERT(strcmp(config.encoder_name, "h264_nvenc") == 0);
    TEST_ASSERT(config.task == PIPELINE_TASK_DETECT);
    TEST_ASSERT(config.mode == PROCESS_GPU);
    TEST_ASSERT(config.filter == FILTER_BLUR_13X13);
    TEST_ASSERT(config.max_frames == 12);
    TEST_ASSERT(config.lossless_output == 1);
    TEST_ASSERT(config.memory_profile == MEMORY_PROFILE_MANUAL);
    TEST_ASSERT(config.memory_budget_mb == 512);
    TEST_ASSERT(config.frame_slots == 8);
    TEST_ASSERT(config.decoder_threads == 3);
    TEST_ASSERT(config.encoder_threads == 5);
    TEST_ASSERT(config.processor_workers == 6);
    TEST_ASSERT(config.enable_benchmark == 0);
    TEST_ASSERT(config.confidence_threshold > 0.34f && config.confidence_threshold < 0.36f);
    TEST_ASSERT(config.iou_threshold > 0.49f && config.iou_threshold < 0.51f);
    TEST_ASSERT(config.inference_input_size == 512);
    TEST_ASSERT(strcmp(config.inference_backend, "tensorrt") == 0);
    TEST_ASSERT(strcmp(config.inference_precision, "fp16") == 0);
    TEST_ASSERT(config.progress_interval == 42);
    TEST_ASSERT(strcmp(config.ffmpeg_log_level, "error") == 0);
    return 0;
}

static int pipeline_config_test_parse_fp32_precision(void) {
    char *argv[] = {
        "VideoComputePipeline",
        "--precision", "fp32"
    };
    const int argc = (int)(sizeof(argv) / sizeof(argv[0]));

    PipelineConfig config;
    pipeline_config_default /* module: pipeline/pipeline_config */ (&config);

    TEST_ASSERT(pipeline_config_parse_args /* module: pipeline/pipeline_config */ (&config, argc, argv) == 0);
    TEST_ASSERT(strcmp(config.inference_precision, "fp32") == 0);
    return 0;
}

static int pipeline_config_test_no_progress(void) {
    char *argv[] = {
        "VideoComputePipeline",
        "--no-progress"
    };
    const int argc = (int)(sizeof(argv) / sizeof(argv[0]));

    PipelineConfig config;
    pipeline_config_default /* module: pipeline/pipeline_config */ (&config);

    TEST_ASSERT(pipeline_config_parse_args /* module: pipeline/pipeline_config */ (&config, argc, argv) == 0);
    TEST_ASSERT(config.progress_interval == 0);
    return 0;
}

static int pipeline_config_test_parse_error_message(void) {
    char *argv[] = {
        "VideoComputePipeline",
        "--confidence", "2"
    };
    const int argc = (int)(sizeof(argv) / sizeof(argv[0]));

    PipelineConfig config;
    pipeline_config_default /* module: pipeline/pipeline_config */ (&config);

    TEST_ASSERT(pipeline_config_parse_args /* module: pipeline/pipeline_config */ (&config, argc, argv) != 0);
    TEST_ASSERT(strstr(pipeline_config_last_error /* module: pipeline/pipeline_config */ (), "--confidence must be between 0 and 1") != NULL);
    return 0;
}

static int pipeline_config_test_missing_value_error_message(void) {
    char *argv[] = {
        "VideoComputePipeline",
        "--input-size"
    };
    const int argc = (int)(sizeof(argv) / sizeof(argv[0]));

    PipelineConfig config;
    pipeline_config_default /* module: pipeline/pipeline_config */ (&config);

    TEST_ASSERT(pipeline_config_parse_args /* module: pipeline/pipeline_config */ (&config, argc, argv) != 0);
    TEST_ASSERT(strstr(pipeline_config_last_error /* module: pipeline/pipeline_config */ (), "missing value for --input-size") != NULL);
    return 0;
}

static int pipeline_config_test_detect_summary(void) {
    PipelineConfig config;
    char summary[4096];

    pipeline_config_default /* module: pipeline/pipeline_config */ (&config);
    config.task = PIPELINE_TASK_DETECT;

    TEST_ASSERT(pipeline_config_format_summary /* module: pipeline/pipeline_config */ (&config, summary, sizeof(summary)) == 0);
    TEST_ASSERT(strstr(summary, "task: detect") != NULL);
    TEST_ASSERT(strstr(summary, "model_path:") != NULL);
    TEST_ASSERT(strstr(summary, "labels_path:") != NULL);
    TEST_ASSERT(strstr(summary, "detections_path:") != NULL);
    TEST_ASSERT(strstr(summary, "confidence:") != NULL);
    TEST_ASSERT(strstr(summary, "input_size:") != NULL);
    TEST_ASSERT(strstr(summary, "filter:") == NULL);
    TEST_ASSERT(strstr(summary, "encoder:") == NULL);
    TEST_ASSERT(strstr(summary, "output_path:") == NULL);
    TEST_ASSERT(strstr(summary, "lossless_output:") == NULL);
    return 0;
}

static int pipeline_config_test_filter_summary(void) {
    PipelineConfig config;
    char summary[4096];

    pipeline_config_default /* module: pipeline/pipeline_config */ (&config);

    TEST_ASSERT(pipeline_config_format_summary /* module: pipeline/pipeline_config */ (&config, summary, sizeof(summary)) == 0);
    TEST_ASSERT(strstr(summary, "task: filter") != NULL);
    TEST_ASSERT(strstr(summary, "output_path:") != NULL);
    TEST_ASSERT(strstr(summary, "encoder:") != NULL);
    TEST_ASSERT(strstr(summary, "mode:") != NULL);
    TEST_ASSERT(strstr(summary, "filter:") != NULL);
    TEST_ASSERT(strstr(summary, "model_path:") == NULL);
    TEST_ASSERT(strstr(summary, "confidence:") == NULL);
    TEST_ASSERT(strstr(summary, "detections_path:") == NULL);
    return 0;
}

int main(void) {
    printf("Running pipeline_config tests...\n");
    if (pipeline_config_test_defaults() != 0) {
        return 1;
    }
    if (pipeline_config_test_parse_args() != 0) {
        return 1;
    }
    if (pipeline_config_test_parse_fp32_precision() != 0) {
        return 1;
    }
    if (pipeline_config_test_no_progress() != 0) {
        return 1;
    }
    if (pipeline_config_test_parse_error_message() != 0) {
        return 1;
    }
    if (pipeline_config_test_missing_value_error_message() != 0) {
        return 1;
    }
    if (pipeline_config_test_detect_summary() != 0) {
        return 1;
    }
    if (pipeline_config_test_filter_summary() != 0) {
        return 1;
    }
    return 0;
}
