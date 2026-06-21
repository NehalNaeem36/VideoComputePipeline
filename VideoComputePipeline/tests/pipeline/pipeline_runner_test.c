#include "pipeline/pipeline_runner.h"
#include "utils/file_utils.h"

#include <stdio.h>
#include <string.h>

#define TEST_ASSERT(expr) \
    do { \
        if (!(expr)) { \
            fprintf(stderr, "Assertion failed: %s:%d: %s\n", __FILE__, __LINE__, #expr); \
            return 1; \
        } \
    } while (0)

int main(void) {
    printf("Running pipeline_runner tests...\n");

#ifndef VCP_SOURCE_DIR
#define VCP_SOURCE_DIR "."
#endif

    PipelineConfig config;
    pipeline_config_default /* module: pipeline/pipeline_config */ (&config);
    strcpy(config.input_path, VCP_SOURCE_DIR "/data/input/15592600_3840_2160_60fps.mp4");
    strcpy(config.output_path, VCP_SOURCE_DIR "/data/output/pipeline_runner_test.mp4");
    strcpy(config.benchmark_path, VCP_SOURCE_DIR "/benchmarks/pipeline_runner_test.csv");
    config.max_frames = 1;
    config.mode = PROCESS_CPU;
    config.filter = FILTER_GRAYSCALE;

    PipelineConfig detect_config;
    pipeline_config_default /* module: pipeline/pipeline_config */ (&detect_config);
    detect_config.task = PIPELINE_TASK_DETECT;
    detect_config.enable_benchmark = 0;
    TEST_ASSERT(pipeline_run /* module: pipeline/pipeline_runner */ (&detect_config) != 0);

    if (!file_exists /* module: utils/file_utils */ (config.input_path)) {
        printf("pipeline_runner_test skipped: input video not available\n");
        return 0;
    }

    TEST_ASSERT(pipeline_run /* module: pipeline/pipeline_runner */ (&config) == 0);
    return 0;
}
