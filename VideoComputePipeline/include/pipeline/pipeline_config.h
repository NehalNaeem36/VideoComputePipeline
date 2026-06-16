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
    FILTER_BLUR_9X9 = 3
} FilterType;

typedef struct {
    char input_path[VCP_MAX_PATH_LENGTH];
    char output_path[VCP_MAX_PATH_LENGTH];
    char benchmark_path[VCP_MAX_PATH_LENGTH];
    ProcessMode mode;
    FilterType filter;
    int max_frames;
    int enable_benchmark;
    int frame_slots;
} PipelineConfig;

void pipeline_config_default(PipelineConfig *config);
int pipeline_config_parse_args(PipelineConfig *config, int argc, char **argv);
void pipeline_config_print(const PipelineConfig *config);

#endif
