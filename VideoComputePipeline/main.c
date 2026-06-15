#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "config.h"
#include "pipeline/pipeline_config.h"
#include "pipeline/pipeline_runner.h"
#include "utils/logger.h"

void print_usage(const char *program_name) {
    printf("Usage: %s [options]\n", program_name);
    printf("Options:\n");
    printf("  -i, --input FILE       Input video file (default: %s)\n", DEFAULT_INPUT_FILE);
    printf("  -o, --output FILE      Output video file (default: %s)\n", DEFAULT_OUTPUT_FILE);
    printf("  -f, --filter FILTER    Filter type: grayscale, blur3x3, blur5x5, blur9x9 (default: %s)\n", DEFAULT_FILTER_TYPE);
    printf("  -g, --gpu              Use GPU processing\n");
    printf("  -t, --threads NUM      Number of threads (default: %d)\n", DEFAULT_NUM_THREADS);
    printf("  -b, --benchmark        Enable benchmarks\n");
    printf("  -h, --help             Show this help message\n");
    printf("  -v, --version          Show version information\n");
}

void print_version(void) {
    printf("%s version %s\n", PROJECT_NAME, PROJECT_VERSION);
    printf("Built on %s at %s\n", BUILD_DATE, BUILD_TIME);
}

int main(int argc, char **argv) {
    printf("%s - %s\n", PROJECT_NAME, "CPU/GPU Video Processing Pipeline");
    printf("Version: %s\n\n", PROJECT_VERSION);

    // Initialize logger
    logger_init(LOG_INFO, stdout);

    // Create pipeline configuration
    PipelineConfig *config = pipeline_config_create();
    if (!config) {
        logger_error("Failed to create pipeline configuration");
        return EXIT_FAILURE;
    }

    // Load defaults
    pipeline_config_load_defaults(config);

    // Parse command-line arguments
    if (pipeline_config_parse_args(config, argc, argv) != 0) {
        print_usage(argv[0]);
        pipeline_config_destroy(config);
        return EXIT_FAILURE;
    }

    // Validate configuration
    if (pipeline_config_validate(config) != 0) {
        logger_error("Invalid configuration");
        pipeline_config_destroy(config);
        return EXIT_FAILURE;
    }

    // Print configuration
    printf("\n--- Pipeline Configuration ---\n");
    pipeline_config_print(config);
    printf("\n");

    // Create pipeline runner
    PipelineRunner *runner = pipeline_runner_create(config);
    if (!runner) {
        logger_error("Failed to create pipeline runner");
        pipeline_config_destroy(config);
        return EXIT_FAILURE;
    }

    // Run pipeline
    logger_info("Starting pipeline execution...");
    int result = pipeline_runner_run(runner);
    if (result != 0) {
        logger_error("Pipeline execution failed with error code %d", result);
    } else {
        logger_info("Pipeline execution completed successfully");
    }

    // Cleanup
    pipeline_runner_destroy(runner);
    pipeline_config_destroy(config);
    logger_shutdown();

    return result == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
