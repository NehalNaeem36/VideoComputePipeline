/*
 * Backend registry module: reports compile/runtime availability for inference
 * backends and owns runtime/model-extension validation. CLI diagnostics and
 * inference engine creation use it before selecting a backend implementation.
 */
#include "inference/backend_registry.h"

#include <ctype.h>
#include <stdio.h>
#include <string.h>

static const char *path_extension(const char *path) {
    const char *dot = NULL;
    const char *slash = NULL;
    const char *backslash = NULL;

    if (!path) {
        return "";
    }

    dot = strrchr(path, '.');
    slash = strrchr(path, '/');
    backslash = strrchr(path, '\\');
    if (!dot || (slash && dot < slash) || (backslash && dot < backslash)) {
        return "";
    }
    return dot;
}

static int str_ieq(const char *a, const char *b) {
    if (!a || !b) {
        return 0;
    }
    while (*a && *b) {
        if (tolower((unsigned char)*a) != tolower((unsigned char)*b)) {
            return 0;
        }
        ++a;
        ++b;
    }
    return *a == '\0' && *b == '\0';
}

const char *inference_runtime_to_string(InferenceRuntime runtime) {
    switch (runtime) {
        case INFERENCE_RUNTIME_TENSORRT: return "tensorrt";
        case INFERENCE_RUNTIME_ONNXRUNTIME: return "onnxruntime";
        case INFERENCE_RUNTIME_TORCHSCRIPT: return "torchscript";
        case INFERENCE_RUNTIME_AUTO:
        default: return "auto";
    }
}

const char *backend_device_to_string(BackendDevice device) {
    return device == BACKEND_DEVICE_CPU ? "cpu" : "cuda";
}

const char *model_type_to_string(ModelType model_type) {
    return model_type == MODEL_TYPE_YOLOV5 ? "yolov5" : "auto";
}

int inference_runtime_parse(const char *value, InferenceRuntime *runtime) {
    if (!value || !runtime) {
        return -1;
    }
    if (strcmp(value, "auto") == 0) {
        *runtime = INFERENCE_RUNTIME_AUTO;
    } else if (strcmp(value, "tensorrt") == 0) {
        *runtime = INFERENCE_RUNTIME_TENSORRT;
    } else if (strcmp(value, "onnxruntime") == 0 || strcmp(value, "onnx") == 0) {
        *runtime = INFERENCE_RUNTIME_ONNXRUNTIME;
    } else if (strcmp(value, "torchscript") == 0 || strcmp(value, "torch") == 0 || strcmp(value, "libtorch") == 0) {
        *runtime = INFERENCE_RUNTIME_TORCHSCRIPT;
    } else {
        return -1;
    }
    return 0;
}

int backend_device_parse(const char *value, BackendDevice *device) {
    if (!value || !device) {
        return -1;
    }
    if (strcmp(value, "cuda") == 0) {
        *device = BACKEND_DEVICE_CUDA;
    } else if (strcmp(value, "cpu") == 0) {
        *device = BACKEND_DEVICE_CPU;
    } else {
        return -1;
    }
    return 0;
}

int model_type_parse(const char *value, ModelType *model_type) {
    if (!value || !model_type) {
        return -1;
    }
    if (strcmp(value, "auto") == 0) {
        *model_type = MODEL_TYPE_AUTO;
    } else if (strcmp(value, "yolov5") == 0) {
        *model_type = MODEL_TYPE_YOLOV5;
    } else {
        return -1;
    }
    return 0;
}

InferenceRuntime inference_runtime_from_model_path(const char *model_path) {
    const char *ext = path_extension(model_path);
    if (str_ieq(ext, ".engine") || str_ieq(ext, ".plan")) {
        return INFERENCE_RUNTIME_TENSORRT;
    }
    if (str_ieq(ext, ".onnx")) {
        return INFERENCE_RUNTIME_ONNXRUNTIME;
    }
    if (str_ieq(ext, ".pt") || str_ieq(ext, ".ts") || str_ieq(ext, ".torchscript")) {
        return INFERENCE_RUNTIME_TORCHSCRIPT;
    }
    return INFERENCE_RUNTIME_AUTO;
}

const char *inference_model_format_from_path(const char *model_path) {
    const InferenceRuntime runtime = inference_runtime_from_model_path(model_path);
    switch (runtime) {
        case INFERENCE_RUNTIME_TENSORRT: return "engine";
        case INFERENCE_RUNTIME_ONNXRUNTIME: return "onnx";
        case INFERENCE_RUNTIME_TORCHSCRIPT: return "torchscript";
        case INFERENCE_RUNTIME_AUTO:
        default: return "unknown";
    }
}

int inference_runtime_validate_model_path(InferenceRuntime runtime,
                                          const char *model_path,
                                          char *message,
                                          size_t message_size) {
    InferenceRuntime detected = INFERENCE_RUNTIME_AUTO;

    if (!model_path || model_path[0] == '\0') {
        snprintf(message, message_size, "model path is empty");
        return -1;
    }

    detected = inference_runtime_from_model_path(model_path);
    if (detected == INFERENCE_RUNTIME_AUTO) {
        snprintf(message,
                 message_size,
                 "unsupported model extension. Supported: .engine, .plan, .onnx, .pt, .ts, .torchscript");
        return -1;
    }

    if (runtime == INFERENCE_RUNTIME_AUTO) {
        return 0;
    }

    if (runtime != detected) {
        snprintf(message,
                 message_size,
                 "model extension selects %s but --runtime is %s",
                 inference_runtime_to_string(detected),
                 inference_runtime_to_string(runtime));
        return -1;
    }

    return 0;
}

int inference_backend_registry_get(InferenceRuntime runtime, BackendRegistryInfo *info) {
    if (!info) {
        return -1;
    }

    memset(info, 0, sizeof(*info));
    info->runtime = runtime;

    switch (runtime) {
        case INFERENCE_RUNTIME_TENSORRT:
            info->compiled =
#ifdef VCP_ENABLE_TENSORRT
                1;
#else
                0;
#endif
            info->available = info->compiled;
            info->cuda_available = info->compiled;
            info->cpu_available = 0;
            info->gpu_io_available = info->compiled;
            info->supports_true_batch = 0;
            info->supports_multi_context = info->compiled;
            snprintf(info->version, sizeof(info->version), "%s", info->compiled ? "linked" : "not compiled");
            snprintf(info->diagnostic,
                     sizeof(info->diagnostic),
                     "%s",
                     info->compiled ? "TensorRT backend is compiled" : "TensorRT backend was not compiled. Rebuild with ENABLE_TENSORRT=ON.");
            return 0;
        case INFERENCE_RUNTIME_ONNXRUNTIME:
            info->compiled =
#ifdef VCP_ENABLE_ONNXRUNTIME
                1;
#else
                0;
#endif
            info->available = info->compiled;
            info->cuda_available = info->compiled;
            info->cpu_available = info->compiled;
            info->gpu_io_available = info->compiled;
            info->supports_true_batch = info->compiled;
            info->supports_multi_context = 0;
            snprintf(info->version, sizeof(info->version), "%s", info->compiled ? "linked" : "not compiled");
            snprintf(info->diagnostic,
                     sizeof(info->diagnostic),
                     "%s",
                     info->compiled ? "ONNX Runtime backend is compiled with CUDA I/O binding support" : "ONNX Runtime backend was not compiled. Rebuild with ENABLE_ONNXRUNTIME=ON.");
            return 0;
        case INFERENCE_RUNTIME_TORCHSCRIPT:
            info->compiled =
#ifdef VCP_ENABLE_LIBTORCH
                1;
#else
                0;
#endif
            info->available = info->compiled;
            info->cuda_available = info->compiled;
            info->cpu_available = info->compiled;
            info->gpu_io_available = info->compiled;
            info->supports_true_batch = info->compiled;
            info->supports_multi_context = 0;
            snprintf(info->version, sizeof(info->version), "%s", info->compiled ? "linked" : "not compiled");
            snprintf(info->diagnostic,
                     sizeof(info->diagnostic),
                     "%s",
                     info->compiled ? "TorchScript backend is compiled with CUDA tensor support" : "TorchScript backend was not compiled. Rebuild with ENABLE_LIBTORCH=ON.");
            return 0;
        case INFERENCE_RUNTIME_AUTO:
        default:
            return -1;
    }
}

void inference_backend_registry_print(FILE *out) {
    const InferenceRuntime runtimes[] = {
        INFERENCE_RUNTIME_TENSORRT,
        INFERENCE_RUNTIME_ONNXRUNTIME,
        INFERENCE_RUNTIME_TORCHSCRIPT,
    };
    FILE *target = out ? out : stdout;

    fprintf(target, "Inference backends:\n");
    for (size_t i = 0; i < sizeof(runtimes) / sizeof(runtimes[0]); ++i) {
        BackendRegistryInfo info;
        inference_backend_registry_get(runtimes[i], &info);
        fprintf(target, "%s:\n", inference_runtime_to_string(runtimes[i]));
        fprintf(target, "  compiled: %s\n", info.compiled ? "yes" : "no");
        fprintf(target, "  available: %s\n", info.available ? "yes" : "no");
        fprintf(target, "  cuda: %s\n", info.cuda_available ? "yes" : "no");
        fprintf(target, "  cpu: %s\n", info.cpu_available ? "yes" : "no");
        fprintf(target, "  gpu_io: %s\n", info.gpu_io_available ? "yes" : "no");
        fprintf(target, "  true_batch: %s\n", info.supports_true_batch ? "yes" : "no");
        fprintf(target, "  multi_context: %s\n", info.supports_multi_context ? "yes" : "no");
        fprintf(target, "  version: %s\n", info.version);
        fprintf(target, "  diagnostic: %s\n", info.diagnostic);
    }
}

void inference_backend_model_info_print(FILE *out,
                                        const char *model_path,
                                        const char *labels_path,
                                        InferenceRuntime runtime,
                                        BackendDevice backend_device,
                                        ModelType model_type) {
    FILE *target = out ? out : stdout;
    InferenceRuntime selected = runtime;
    char validation[256] = {0};
    BackendRegistryInfo info;

    if (selected == INFERENCE_RUNTIME_AUTO) {
        selected = inference_runtime_from_model_path(model_path);
    }

    fprintf(target, "Model info:\n");
    fprintf(target, "  model_path: %s\n", model_path ? model_path : "");
    fprintf(target, "  detected_format: %s\n", inference_model_format_from_path(model_path));
    fprintf(target, "  selected_runtime: %s\n", inference_runtime_to_string(selected));
    fprintf(target, "  backend_device: %s\n", backend_device_to_string(backend_device));
    fprintf(target, "  model_adapter: %s\n", model_type == MODEL_TYPE_AUTO ? "yolov5 (auto default)" : model_type_to_string(model_type));
    fprintf(target, "  labels_path: %s\n", labels_path ? labels_path : "");

    if (inference_runtime_validate_model_path(runtime, model_path, validation, sizeof(validation)) != 0) {
        fprintf(target, "  valid: no\n");
        fprintf(target, "  warning: %s\n", validation);
        return;
    }

    if (inference_backend_registry_get(selected, &info) == 0) {
        fprintf(target, "  valid: yes\n");
        fprintf(target, "  backend_compiled: %s\n", info.compiled ? "yes" : "no");
        fprintf(target, "  backend_available: %s\n", info.available ? "yes" : "no");
        fprintf(target, "  gpu_resident_possible: %s\n", info.gpu_io_available && backend_device == BACKEND_DEVICE_CUDA ? "yes" : "no");
        fprintf(target, "  backend_diagnostic: %s\n", info.diagnostic);
    }
}
