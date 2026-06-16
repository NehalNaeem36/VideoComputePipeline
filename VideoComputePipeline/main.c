#include "config.h"
#include "pipeline/pipeline_config.h"
#include "pipeline/pipeline_runner.h"
#include "utils/logger.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void print_usage(const char *program_name) {
    printf("Usage: %s [options]\n", program_name);
    printf("Options:\n");
    printf("  --input path              Input MP4 path\n");
    printf("  --output path             Output MP4 path\n");
    printf("  --benchmark path          Benchmark CSV path\n");
    printf("  --mode cpu|gpu            Processing mode\n");
    printf("  --filter grayscale|blur3x3|blur5x5|blur9x9\n");
    printf("  --max-frames N            Stop after N frames, 0 means no limit\n");
    printf("  --frame-slots N           Frames buffered between pipeline stages\n");
    printf("  --decoder-threads N       FFmpeg decoder worker threads\n");
    printf("  --encoder-threads N       FFmpeg encoder worker threads\n");
    printf("  --processor-workers N     CPU processor workers, GPU mode uses one worker\n");
    printf("  --no-benchmark            Disable benchmark output\n");
    printf("  --help                    Show this help message\n");
    printf("  --version                 Show version information\n");
}

static void print_version(void) {
    printf("%s %s\n", PROJECT_NAME, PROJECT_VERSION);
}

int main(int argc, char **argv) {
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return EXIT_SUCCESS;
        }

        if (strcmp(argv[i], "--version") == 0) {
            print_version();
            return EXIT_SUCCESS;
        }
    }

    PipelineConfig config;
    pipeline_config_default(&config);

    if (pipeline_config_parse_args(&config, argc, argv) != 0) {
        log_error("invalid command-line arguments");
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    printf("%s %s\n", PROJECT_NAME, PROJECT_VERSION);
    printf("configuration:\n");
    pipeline_config_print(&config);

    log_info("starting pipeline");
    const int result = pipeline_run(&config);
    if (result != 0) {
        log_error("pipeline failed with code %d", result);
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
