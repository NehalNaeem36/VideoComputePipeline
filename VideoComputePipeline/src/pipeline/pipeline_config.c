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
        case FILTER_BLUR_13X13:
            return "blur13x13";
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

    if (strcmp(value, "blur13x13") == 0) {
        *filter = FILTER_BLUR_13X13;
        return 0;
    }

    return -1;
}

static const char *memory_profile_to_string(MemoryProfile profile) {
    switch (profile) {
        case MEMORY_PROFILE_AUTO:
            return "auto";
        case MEMORY_PROFILE_LOW:
            return "low";
        case MEMORY_PROFILE_BALANCED:
            return "balanced";
        case MEMORY_PROFILE_MANUAL:
            return "manual";
        default:
            return "unknown";
    }
}

static int parse_memory_profile(const char *value, MemoryProfile *profile) {
    if (strcmp(value, "auto") == 0) {
        *profile = MEMORY_PROFILE_AUTO;
        return 0;
    }
    if (strcmp(value, "low") == 0) {
        *profile = MEMORY_PROFILE_LOW;
        return 0;
    }
    if (strcmp(value, "balanced") == 0) {
        *profile = MEMORY_PROFILE_BALANCED;
        return 0;
    }
    if (strcmp(value, "manual") == 0) {
        *profile = MEMORY_PROFILE_MANUAL;
        return 0;
    }
    return -1;
}

void pipeline_config_default(PipelineConfig *config) {
    if (!config) {
        return;
    }

    copy_path /* module: pipeline/pipeline_config */ (config->input_path, sizeof(config->input_path), DEFAULT_INPUT_PATH);
    copy_path /* module: pipeline/pipeline_config */ (config->output_path, sizeof(config->output_path), DEFAULT_OUTPUT_PATH);
    copy_path /* module: pipeline/pipeline_config */ (config->benchmark_path, sizeof(config->benchmark_path), DEFAULT_BENCHMARK_PATH);
    copy_path /* module: pipeline/pipeline_config */ (config->encoder_name, sizeof(config->encoder_name), DEFAULT_ENCODER_NAME);
    config->mode = PROCESS_CPU;
    config->filter = FILTER_GRAYSCALE;
    config->max_frames = DEFAULT_MAX_FRAMES;
    config->enable_benchmark = DEFAULT_ENABLE_BENCHMARK;
    config->lossless_output = DEFAULT_LOSSLESS_OUTPUT;
    config->memory_profile = MEMORY_PROFILE_AUTO;
    config->memory_budget_mb = DEFAULT_MEMORY_BUDGET_MB;
    config->frame_slots = DEFAULT_FRAME_SLOTS;
    config->decoder_threads = DEFAULT_DECODER_THREADS;
    config->encoder_threads = DEFAULT_ENCODER_THREADS;
    config->processor_workers = DEFAULT_PROCESSOR_WORKERS;
}

int pipeline_config_parse_args(PipelineConfig *config, int argc, char **argv) {
    if (!config) {
        return -1;
    }

    for (int i = 1; i < argc; ++i) {
        const char *arg = argv[i];

        if (strcmp(arg, "--input") == 0 && i + 1 < argc) {
            if (copy_path /* module: pipeline/pipeline_config */ (config->input_path, sizeof(config->input_path), argv[++i]) != 0) {
                return -1;
            }
        } else if (strcmp(arg, "--output") == 0 && i + 1 < argc) {
            if (copy_path /* module: pipeline/pipeline_config */ (config->output_path, sizeof(config->output_path), argv[++i]) != 0) {
                return -1;
            }
        } else if (strcmp(arg, "--benchmark") == 0 && i + 1 < argc) {
            if (copy_path /* module: pipeline/pipeline_config */ (config->benchmark_path, sizeof(config->benchmark_path), argv[++i]) != 0) {
                return -1;
            }
        } else if (strcmp(arg, "--encoder") == 0 && i + 1 < argc) {
            const char *encoder = argv[++i];
            if (strcmp(encoder, "libx264") != 0 &&
                strcmp(encoder, "libx264rgb") != 0 &&
                strcmp(encoder, "h264_nvenc") != 0 &&
                strcmp(encoder, "mpeg4") != 0) {
                return -1;
            }
            if (copy_path /* module: pipeline/pipeline_config */ (config->encoder_name, sizeof(config->encoder_name), encoder) != 0) {
                return -1;
            }
        } else if (strcmp(arg, "--mode") == 0 && i + 1 < argc) {
            if (parse_mode /* module: pipeline/pipeline_config */ (argv[++i], &config->mode) != 0) {
                return -1;
            }
        } else if (strcmp(arg, "--filter") == 0 && i + 1 < argc) {
            if (parse_filter /* module: pipeline/pipeline_config */ (argv[++i], &config->filter) != 0) {
                return -1;
            }
        } else if (strcmp(arg, "--max-frames") == 0 && i + 1 < argc) {
            config->max_frames = atoi(argv[++i]);
            if (config->max_frames < 0) {
                return -1;
            }
        } else if (strcmp(arg, "--frame-slots") == 0 && i + 1 < argc) {
            config->frame_slots = atoi(argv[++i]);
            if (config->frame_slots <= 0) {
                return -1;
            }
        } else if (strcmp(arg, "--decoder-threads") == 0 && i + 1 < argc) {
            config->decoder_threads = atoi(argv[++i]);
            if (config->decoder_threads <= 0) {
                return -1;
            }
        } else if (strcmp(arg, "--encoder-threads") == 0 && i + 1 < argc) {
            config->encoder_threads = atoi(argv[++i]);
            if (config->encoder_threads <= 0) {
                return -1;
            }
        } else if (strcmp(arg, "--processor-workers") == 0 && i + 1 < argc) {
            config->processor_workers = atoi(argv[++i]);
            if (config->processor_workers <= 0) {
                return -1;
            }
        } else if (strcmp(arg, "--no-benchmark") == 0) {
            config->enable_benchmark = 0;
        } else if (strcmp(arg, "--lossless") == 0) {
            config->lossless_output = 1;
        } else if (strcmp(arg, "--lossy") == 0) {
            config->lossless_output = 0;
        } else if (strcmp(arg, "--memory-profile") == 0 && i + 1 < argc) {
            if (parse_memory_profile /* module: pipeline/pipeline_config */ (argv[++i], &config->memory_profile) != 0) {
                return -1;
            }
        } else if (strcmp(arg, "--memory-budget-mb") == 0 && i + 1 < argc) {
            config->memory_budget_mb = atoi(argv[++i]);
            if (config->memory_budget_mb < 0) {
                return -1;
            }
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
    printf("encoder: %s\n", config->encoder_name);
    printf("mode: %s\n", mode_to_string(config->mode));
    printf("filter: %s\n", filter_to_string(config->filter));
    printf("max_frames: %d\n", config->max_frames);
    printf("enable_benchmark: %s\n", config->enable_benchmark ? "true" : "false");
    printf("lossless_output: %s\n", config->lossless_output ? "true" : "false");
    printf("memory_profile: %s\n", memory_profile_to_string(config->memory_profile));
    printf("memory_budget_mb: %d\n", config->memory_budget_mb);
    printf("frame_slots: %d\n", config->frame_slots);
    printf("decoder_threads: %d\n", config->decoder_threads);
    printf("encoder_threads: %d\n", config->encoder_threads);
    printf("processor_workers: %d\n", config->processor_workers);
}
