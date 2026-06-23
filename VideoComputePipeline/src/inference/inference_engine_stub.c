/*
 * Inference engine stub module: satisfies the inference C API when CUDA/TensorRT
 * support is disabled. Detection mode can fail cleanly at runtime while non-CUDA
 * builds continue to compile and run filter paths.
 */
#include "inference/inference_engine.h"
#include "inference/backend_registry.h"

#include <stdlib.h>

struct InferenceEngine {
    int unused;
};

static const char *g_last_error = "CUDA/TensorRT inference backend was not built";

int inference_engine_create(InferenceEngine **engine, const InferenceConfig *config) {
    InferenceRuntime runtime = INFERENCE_RUNTIME_TENSORRT;
    if (engine) {
        *engine = NULL;
    }
    if (config) {
        runtime = config->runtime == INFERENCE_RUNTIME_AUTO
                      ? inference_runtime_from_model_path(config->model_path)
                      : config->runtime;
    }
    if (runtime == INFERENCE_RUNTIME_ONNXRUNTIME) {
        g_last_error = "ONNX Runtime backend was not compiled. Rebuild with ENABLE_ONNXRUNTIME=ON.";
    } else if (runtime == INFERENCE_RUNTIME_TORCHSCRIPT) {
        g_last_error = "TorchScript backend was not compiled. Rebuild with ENABLE_LIBTORCH=ON.";
    } else {
        g_last_error = "TensorRT backend was not compiled. Rebuild with ENABLE_CUDA_INFERENCE=ON and ENABLE_TENSORRT=ON.";
    }
    return -1;
}

int inference_engine_get_batch_capability(InferenceEngine *engine, InferenceBatchCapability *capability) {
    (void)engine;
    if (capability) {
        capability->min_batch_size = 1;
        capability->max_batch_size = 1;
        capability->supports_dynamic_batch = 0;
        capability->supports_true_batching = 0;
        capability->supports_parallel_contexts = 0;
        capability->max_parallel_contexts = 1;
        capability->selected_context_count = 1;
        capability->input_width = 0;
        capability->input_height = 0;
        capability->input_bytes_per_frame = 0;
        capability->output_bytes_per_frame = 0;
        capability->description[0] = '\0';
    }
    g_last_error = "CUDA/TensorRT inference backend was not built";
    return -1;
}

int inference_engine_set_parallel_contexts(InferenceEngine *engine, int context_count) {
    (void)engine;
    (void)context_count;
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

int inference_engine_run_nv12_batch(InferenceEngine *engine, FrameBatch *batch) {
    (void)engine;
    (void)batch;
    g_last_error = "CUDA/TensorRT inference backend was not built";
    return -1;
}

int inference_engine_run_cuda_nv12_batch(InferenceEngine *engine, FrameBatch *batch) {
    (void)engine;
    (void)batch;
    g_last_error = "CUDA/TensorRT inference backend was not built";
    return -1;
}

void inference_engine_destroy(InferenceEngine *engine) {
    free(engine);
}

const char *inference_engine_last_error(void) {
    return g_last_error;
}
