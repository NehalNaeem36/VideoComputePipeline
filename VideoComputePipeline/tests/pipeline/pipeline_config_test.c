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
    TEST_ASSERT(strcmp(config.encoder_name, DEFAULT_ENCODER_NAME) == 0);
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
    return 0;
}

static int pipeline_config_test_parse_args(void) {
    char *argv[] = {
        "VideoComputePipeline",
        "--input", "input.mp4",
        "--output", "output.mp4",
        "--benchmark", "bench.csv",
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
        "--no-benchmark"
    };
    const int argc = (int)(sizeof(argv) / sizeof(argv[0]));

    PipelineConfig config;
    pipeline_config_default /* module: pipeline/pipeline_config */ (&config);

    TEST_ASSERT(pipeline_config_parse_args /* module: pipeline/pipeline_config */ (&config, argc, argv) == 0);
    TEST_ASSERT(strcmp(config.input_path, "input.mp4") == 0);
    TEST_ASSERT(strcmp(config.output_path, "output.mp4") == 0);
    TEST_ASSERT(strcmp(config.benchmark_path, "bench.csv") == 0);
    TEST_ASSERT(strcmp(config.encoder_name, "h264_nvenc") == 0);
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
    return 0;
}
