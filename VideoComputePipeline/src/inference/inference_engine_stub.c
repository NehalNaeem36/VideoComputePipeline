#include "inference/inference_engine.h"

#include <stdlib.h>

struct InferenceEngine {
    int unused;
};

static const char *g_last_error = "CUDA/TensorRT inference backend was not built";

int inference_engine_create(InferenceEngine **engine, const InferenceConfig *config) {
    (void)config;
    if (engine) {
        *engine = NULL;
    }
    g_last_error = "CUDA/TensorRT inference backend was not built";
    return -1;
}

int inference_engine_run_nv12(InferenceEngine *engine, const Frame *frame, DetectionResult *result, FrameTiming *timing) {
    (void)engine;
    (void)frame;
    (void)result;
    (void)timing;
    g_last_error = "CUDA/TensorRT inference backend was not built";
    return -1;
}

int inference_engine_run_cuda_nv12(InferenceEngine *engine, const CudaNV12Frame *frame, DetectionResult *result, FrameTiming *timing) {
    (void)engine;
    (void)frame;
    (void)result;
    (void)timing;
    g_last_error = "CUDA/TensorRT inference backend was not built";
    return -1;
}

void inference_engine_destroy(InferenceEngine *engine) {
    free(engine);
}

const char *inference_engine_last_error(void) {
    return g_last_error;
}
