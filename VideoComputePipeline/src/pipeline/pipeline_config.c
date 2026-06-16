#include "pipeline/pipeline_config.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int copy_path(char *dst, size_t dst_size, const char *src) {
    if (!dst || !src || strlen(src) >= dst_size) {
        return -1;
    }

    strcpy(dst, src);
    return 0;
}

static const char *mode_to_string(ProcessMode mode) {
    return mode == PROCESS_GPU ? "gpu" : "cpu";
}

static const char *filter_to_string(FilterType filter) {
    switch (filter) {
        case FILTER_GRAYSCALE:
            return "grayscale";
        case FILTER_BLUR_3X3:
            return "blur3x3";
        case FILTER_BLUR_5X5:
            return "blur5x5";
        case FILTER_BLUR_9X9:
            return "blur9x9";
        default:
            return "unknown";
    }
}

static int parse_mode(const char *value, ProcessMode *mode) {
    if (strcmp(value, "cpu") == 0) {
        *mode = PROCESS_CPU;
        return 0;
    }

    if (strcmp(value, "gpu") == 0) {
        *mode = PROCESS_GPU;
        return 0;
    }

    return -1;
}

static int parse_filter(const char *value, FilterType *filter) {
    if (strcmp(value, "grayscale") == 0) {
        *filter = FILTER_GRAYSCALE;
        return 0;
    }

    if (strcmp(value, "blur3x3") == 0) {
        *filter = FILTER_BLUR_3X3;
        return 0;
    }

    if (strcmp(value, "blur5x5") == 0) {
        *filter = FILTER_BLUR_5X5;
        return 0;
    }

    if (strcmp(value, "blur9x9") == 0) {
        *filter = FILTER_BLUR_9X9;
        return 0;
    }

    return -1;
}

void pipeline_config_default(PipelineConfig *config) {
    if (!config) {
        return;
    }

    copy_path(config->input_path, sizeof(config->input_path), DEFAULT_INPUT_PATH);
    copy_path(config->output_path, sizeof(config->output_path), DEFAULT_OUTPUT_PATH);
    copy_path(config->benchmark_path, sizeof(config->benchmark_path), DEFAULT_BENCHMARK_PATH);
    config->mode = PROCESS_CPU;
    config->filter = FILTER_GRAYSCALE;
    config->max_frames = DEFAULT_MAX_FRAMES;
    config->enable_benchmark = DEFAULT_ENABLE_BENCHMARK;
    config->frame_slots = DEFAULT_FRAME_SLOTS;
}

int pipeline_config_parse_args(PipelineConfig *config, int argc, char **argv) {
    if (!config) {
        return -1;
    }

    for (int i = 1; i < argc; ++i) {
        const char *arg = argv[i];

        if (strcmp(arg, "--input") == 0 && i + 1 < argc) {
            if (copy_path(config->input_path, sizeof(config->input_path), argv[++i]) != 0) {
                return -1;
            }
        } else if (strcmp(arg, "--output") == 0 && i + 1 < argc) {
            if (copy_path(config->output_path, sizeof(config->output_path), argv[++i]) != 0) {
                return -1;
            }
        } else if (strcmp(arg, "--benchmark") == 0 && i + 1 < argc) {
            if (copy_path(config->benchmark_path, sizeof(config->benchmark_path), argv[++i]) != 0) {
                return -1;
            }
        } else if (strcmp(arg, "--mode") == 0 && i + 1 < argc) {
            if (parse_mode(argv[++i], &config->mode) != 0) {
                return -1;
            }
        } else if (strcmp(arg, "--filter") == 0 && i + 1 < argc) {
            if (parse_filter(argv[++i], &config->filter) != 0) {
                return -1;
            }
        } else if (strcmp(arg, "--max-frames") == 0 && i + 1 < argc) {
            config->max_frames = atoi(argv[++i]);
            if (config->max_frames < 0) {
                return -1;
            }
        } else if (strcmp(arg, "--no-benchmark") == 0) {
            config->enable_benchmark = 0;
        } else {
            return -1;
        }
    }

    return 0;
}

void pipeline_config_print(const PipelineConfig *config) {
    if (!config) {
        return;
    }

    printf("input_path: %s\n", config->input_path);
    printf("output_path: %s\n", config->output_path);
    printf("benchmark_path: %s\n", config->benchmark_path);
    printf("mode: %s\n", mode_to_string(config->mode));
    printf("filter: %s\n", filter_to_string(config->filter));
    printf("max_frames: %d\n", config->max_frames);
    printf("enable_benchmark: %s\n", config->enable_benchmark ? "true" : "false");
    printf("frame_slots: %d\n", config->frame_slots);
}
