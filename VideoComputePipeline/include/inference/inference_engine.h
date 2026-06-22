#ifndef VIDEOCOMPUTEPIPELINE_INFERENCE_ENGINE_H
#define VIDEOCOMPUTEPIPELINE_INFERENCE_ENGINE_H

#include "benchmark/benchmark.h"
#include "config.h"
#include "core/frame.h"
#include "gpu/cuda_frame.h"
#include "inference/detection_result.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct InferenceEngine InferenceEngine;

typedef struct {
    char model_path[VCP_MAX_PATH_LENGTH];
    char labels_path[VCP_MAX_PATH_LENGTH];
    int input_width;
    int input_height;
    int class_count;
    int class_filter_id_count;
    int class_filter_ids[VCP_MAX_CLASS_FILTERS];
    int class_filter_name_count;
    char class_filter_names[VCP_MAX_CLASS_FILTERS][VCP_MAX_CLASS_NAME_LENGTH];
    float confidence_threshold;
    float iou_threshold;
    int use_fp16;
} InferenceConfig;

int inference_engine_create(InferenceEngine **engine, const InferenceConfig *config);
int inference_engine_run_nv12(InferenceEngine *engine, const Frame *frame, DetectionResult *result, FrameTiming *timing);
int inference_engine_run_cuda_nv12(InferenceEngine *engine, const CudaNV12Frame *frame, DetectionResult *result, FrameTiming *timing);
void inference_engine_destroy(InferenceEngine *engine);
const char *inference_engine_last_error(void);

#ifdef __cplusplus
}
#endif

#endif
