#include "config.h"
#include "benchmark/matrix_report.h"
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
    printf("  --encoder name            libx264, libx264rgb, h264_nvenc, or mpeg4\n");
    printf("  --mode cpu|gpu            Processing mode\n");
    printf("  --filter grayscale|blur3x3|blur5x5|blur9x9|blur13x13\n");
    printf("  --max-frames N            Stop after N frames, 0 means no limit\n");
    printf("  --frame-slots N           Frames buffered between pipeline stages\n");
    printf("  --decoder-threads N       FFmpeg decoder worker threads\n");
    printf("  --encoder-threads N       FFmpeg encoder worker threads\n");
    printf("  --processor-workers N     CPU processor workers, GPU mode uses one worker\n");
    printf("  --lossless                Use lossless encoder settings when supported\n");
    printf("  --memory-profile name     auto, low, balanced, or manual\n");
    printf("  --memory-budget-mb N      Optional CPU frame-pool budget, 0 means automatic\n");
    printf("  --matrix-report cpu.csv gpu.csv\n");
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
            print_usage /* module: app/main */ (argv[0]);
            return EXIT_SUCCESS;
        }

        if (strcmp(argv[i], "--version") == 0) {
            print_version /* module: app/main */ ();
            return EXIT_SUCCESS;
        }

        if (strcmp(argv[i], "--matrix-report") == 0) {
            if (i + 2 >= argc) {
                log_error /* module: utils/logger */ ("--matrix-report requires CPU and GPU benchmark CSV paths");
                print_usage /* module: app/main */ (argv[0]);
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
    pipeline_config_default /* module: pipeline/pipeline_config */ (&config);

    if (pipeline_config_parse_args /* module: pipeline/pipeline_config */ (&config, argc, argv) != 0) {
        log_error /* module: utils/logger */ ("invalid command-line arguments");
        print_usage /* module: app/main */ (argv[0]);
        return EXIT_FAILURE;
    }

    printf("%s %s\n", PROJECT_NAME, PROJECT_VERSION);
    printf("configuration:\n");
    pipeline_config_print /* module: pipeline/pipeline_config */ (&config);

    log_info /* module: utils/logger */ ("starting pipeline");
    const int result = pipeline_run /* module: pipeline/pipeline_runner */ (&config);
    if (result != 0) {
        log_error /* module: utils/logger */ ("pipeline failed with code %d", result);
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
