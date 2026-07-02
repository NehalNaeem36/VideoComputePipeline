#ifndef VIDEOCOMPUTEPIPELINE_INFERENCE_TYPES_H
#define VIDEOCOMPUTEPIPELINE_INFERENCE_TYPES_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    INFERENCE_RUNTIME_AUTO = 0,
    INFERENCE_RUNTIME_TENSORRT = 1,
    INFERENCE_RUNTIME_ONNXRUNTIME = 2,
    INFERENCE_RUNTIME_TORCHSCRIPT = 3
} InferenceRuntime;

typedef enum {
    BACKEND_DEVICE_CUDA = 0,
    BACKEND_DEVICE_CPU = 1
} BackendDevice;

typedef enum {
    INFERENCE_EXEC_TRUE_BATCH = 0,
    INFERENCE_EXEC_MULTI_CONTEXT = 1,
    INFERENCE_EXEC_SERIAL_BATCH_FALLBACK = 2
} InferenceExecutionMode;

typedef enum {
    MODEL_TYPE_AUTO = 0,
    MODEL_TYPE_YOLOV5 = 1
} ModelType;

typedef enum {
    TENSOR_DTYPE_FP32 = 0,
    TENSOR_DTYPE_FP16 = 1,
    TENSOR_DTYPE_INT32 = 2,
    TENSOR_DTYPE_INT8 = 3,
    TENSOR_DTYPE_UNKNOWN = 255
} TensorDataType;

typedef enum {
    TENSOR_LAYOUT_NCHW = 0,
    TENSOR_LAYOUT_NHWC = 1,
    TENSOR_LAYOUT_UNKNOWN = 255
} TensorLayout;

typedef enum {
    TENSOR_MEMORY_CUDA = 0,
    TENSOR_MEMORY_HOST = 1,
    TENSOR_MEMORY_BACKEND_OWNED = 2,
    TENSOR_MEMORY_UNKNOWN = 255
} TensorMemoryType;

typedef struct {
    void *data;
    int batch_size;
    int channels;
    int height;
    int width;
    TensorDataType dtype;
    TensorLayout layout;
    size_t bytes;
    void *cuda_stream;
    int owns_memory;
    int shape[8];
    int shape_rank;
    char tensor_name[128];
    TensorMemoryType memory_type;
} GpuTensorBatch;

typedef struct {
    int supports_cuda;
    int supports_cpu;
    int supports_gpu_input;
    int supports_gpu_output;
    int supports_true_batch;
    int supports_dynamic_batch;
    int supports_fp16;
    int supports_fp32;
    int supports_external_cuda_memory;
    int requires_host_input;
    int requires_host_output;
    int max_batch_size;
    TensorLayout preferred_input_layout;
    TensorDataType preferred_dtype;
    BackendDevice input_device;
    BackendDevice output_device;
    int can_use_current_cuda_stream;
    char backend_name[64];
    char backend_version[128];
    char diagnostic_message[256];
} InferenceBackendCapabilities;

typedef struct {
    double pre_bind_ms;
    double inference_ms;
    double output_sync_ms;
    double total_backend_ms;
} BackendTiming;

const char *inference_runtime_to_string(InferenceRuntime runtime);
const char *backend_device_to_string(BackendDevice device);
const char *model_type_to_string(ModelType model_type);

int inference_runtime_parse(const char *value, InferenceRuntime *runtime);
int backend_device_parse(const char *value, BackendDevice *device);
int model_type_parse(const char *value, ModelType *model_type);

#ifdef __cplusplus
}
#endif

#endif
