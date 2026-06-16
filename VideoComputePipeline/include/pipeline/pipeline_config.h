#ifndef VIDEOCOMPUTEPIPELINE_PIPELINE_CONFIG_H
#define VIDEOCOMPUTEPIPELINE_PIPELINE_CONFIG_H

#include "config.h"

typedef enum {
    PROCESS_CPU = 0,
    PROCESS_GPU = 1
} ProcessMode;

typedef enum {
    FILTER_GRAYSCALE = 0,
    FILTER_BLUR_3X3 = 1,
    FILTER_BLUR_5X5 = 2,
    FILTER_BLUR_9X9 = 3,
    FILTER_BLUR_13X13 = 4
} FilterType;

typedef enum {
    MEMORY_PROFILE_AUTO = 0,
    MEMORY_PROFILE_LOW = 1,
    MEMORY_PROFILE_BALANCED = 2,
    MEMORY_PROFILE_MANUAL = 3
} MemoryProfile;

typedef struct {
    char input_path[VCP_MAX_PATH_LENGTH];
    char output_path[VCP_MAX_PATH_LENGTH];
    char benchmark_path[VCP_MAX_PATH_LENGTH];
    char encoder_name[64];
    ProcessMode mode;
    FilterType filter;
    int max_frames;
    int enable_benchmark;
    int lossless_output;
    MemoryProfile memory_profile;
    int memory_budget_mb;
    int frame_slots;
    int decoder_threads;
    int encoder_threads;
    int processor_workers;
} PipelineConfig;

void pipeline_config_default(PipelineConfig *config);
int pipeline_config_parse_args(PipelineConfig *config, int argc, char **argv);
void pipeline_config_print(const PipelineConfig *config);

#endif
