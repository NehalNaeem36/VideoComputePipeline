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
    pipeline_config_default(&config);

    TEST_ASSERT(strcmp(config.input_path, DEFAULT_INPUT_PATH) == 0);
    TEST_ASSERT(strcmp(config.output_path, DEFAULT_OUTPUT_PATH) == 0);
    TEST_ASSERT(strcmp(config.benchmark_path, DEFAULT_BENCHMARK_PATH) == 0);
    TEST_ASSERT(config.mode == PROCESS_CPU);
    TEST_ASSERT(config.filter == FILTER_GRAYSCALE);
    TEST_ASSERT(config.max_frames == DEFAULT_MAX_FRAMES);
    TEST_ASSERT(config.enable_benchmark == DEFAULT_ENABLE_BENCHMARK);
    TEST_ASSERT(config.frame_slots == DEFAULT_FRAME_SLOTS);
    return 0;
}

static int pipeline_config_test_parse_args(void) {
    char *argv[] = {
        "VideoComputePipeline",
        "--input", "input.mp4",
        "--output", "output.mp4",
        "--benchmark", "bench.csv",
        "--mode", "gpu",
        "--filter", "blur9x9",
        "--max-frames", "12",
        "--no-benchmark"
    };
    const int argc = (int)(sizeof(argv) / sizeof(argv[0]));

    PipelineConfig config;
    pipeline_config_default(&config);

    TEST_ASSERT(pipeline_config_parse_args(&config, argc, argv) == 0);
    TEST_ASSERT(strcmp(config.input_path, "input.mp4") == 0);
    TEST_ASSERT(strcmp(config.output_path, "output.mp4") == 0);
    TEST_ASSERT(strcmp(config.benchmark_path, "bench.csv") == 0);
    TEST_ASSERT(config.mode == PROCESS_GPU);
    TEST_ASSERT(config.filter == FILTER_BLUR_9X9);
    TEST_ASSERT(config.max_frames == 12);
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
