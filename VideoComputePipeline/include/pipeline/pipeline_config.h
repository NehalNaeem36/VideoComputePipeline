#ifndef VIDEOCOMPUTEPIPELINE_PIPELINE_CONFIG_H
#define VIDEOCOMPUTEPIPELINE_PIPELINE_CONFIG_H

#include "config.h"

#include <stddef.h>

typedef enum {
    PROCESS_CPU = 0,
    PROCESS_GPU = 1
} ProcessMode;

typedef enum {
    PIPELINE_TASK_FILTER = 0,
    PIPELINE_TASK_DETECT = 1
} PipelineTask;

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

typedef enum {
    VIDEO_DECODER_CPU = 0,
    VIDEO_DECODER_NVDEC = 1
} VideoDecoderMode;

typedef enum {
    DECODER_FALLBACK_NONE = 0,
    DECODER_FALLBACK_CPU = 1
} DecoderFallbackMode;

typedef enum {
    OUTPUT_FORMAT_AUTO = 0,
    OUTPUT_FORMAT_MP4 = 1,
    OUTPUT_FORMAT_MKV = 2
} OutputFormat;

typedef struct {
    char input_path[VCP_MAX_PATH_LENGTH];
    char output_path[VCP_MAX_PATH_LENGTH];
    char benchmark_path[VCP_MAX_PATH_LENGTH];
    char detections_path[VCP_MAX_PATH_LENGTH];
    char model_path[VCP_MAX_PATH_LENGTH];
    char labels_path[VCP_MAX_PATH_LENGTH];
    char inference_backend[64];
    char inference_precision[32];
    char encoder_name[64];
    PipelineTask task;
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
    float confidence_threshold;
    float iou_threshold;
    int inference_input_size;
    int detection_class_count;
    int max_detections_per_frame;
    int progress_interval;
    char ffmpeg_log_level[16];
    VideoDecoderMode decoder_mode;
    DecoderFallbackMode decoder_fallback;
    OutputFormat output_format;
    int draw_boxes;
    int box_thickness;
    float box_confidence;
} PipelineConfig;

void pipeline_config_default(PipelineConfig *config);
int pipeline_config_parse_args(PipelineConfig *config, int argc, char **argv);
const char *pipeline_config_last_error(void);
int pipeline_config_format_summary(const PipelineConfig *config, char *buffer, size_t buffer_size);
void pipeline_config_print(const PipelineConfig *config);

#endif
