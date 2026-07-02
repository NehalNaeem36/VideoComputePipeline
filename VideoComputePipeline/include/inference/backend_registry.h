#ifndef VIDEOCOMPUTEPIPELINE_BACKEND_REGISTRY_H
#define VIDEOCOMPUTEPIPELINE_BACKEND_REGISTRY_H

#include "inference/inference_types.h"

#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    InferenceRuntime runtime;
    int compiled;
    int available;
    int cuda_available;
    int cpu_available;
    int gpu_io_available;
    int supports_true_batch;
    int supports_multi_context;
    char version[128];
    char diagnostic[256];
} BackendRegistryInfo;

InferenceRuntime inference_runtime_from_model_path(const char *model_path);
const char *inference_model_format_from_path(const char *model_path);
int inference_runtime_validate_model_path(InferenceRuntime runtime,
                                          const char *model_path,
                                          char *message,
                                          size_t message_size);
int inference_backend_registry_get(InferenceRuntime runtime, BackendRegistryInfo *info);
void inference_backend_registry_print(FILE *out);
void inference_backend_model_info_print(FILE *out,
                                        const char *model_path,
                                        const char *labels_path,
                                        InferenceRuntime runtime,
                                        BackendDevice backend_device,
                                        ModelType model_type);

#ifdef __cplusplus
}
#endif

#endif
