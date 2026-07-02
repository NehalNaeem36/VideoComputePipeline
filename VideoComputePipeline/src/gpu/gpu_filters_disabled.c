/*
 * GPU filters disabled module: keeps CPU-only builds linkable when CUDA GPU
 * filters are intentionally disabled. Pipeline GPU mode fails clearly here.
 */
#include "gpu/gpu_filters.h"
#include "utils/logger.h"

#include <string.h>

static int report_disabled(void) {
    log_error /* module: utils/logger */ ("CUDA GPU filters were not built");
    return -1;
}

int gpu_filters_init(GPUFilterContext *gpu) {
    if (gpu) {
        memset(gpu, 0, sizeof(*gpu));
    }
    return report_disabled /* module: gpu/gpu_filters */ ();
}

void gpu_filters_print_info(const GPUFilterContext *gpu) {
    (void)gpu;
}

void gpu_filters_release(GPUFilterContext *gpu) {
    if (gpu) {
        memset(gpu, 0, sizeof(*gpu));
    }
}

int gpu_grayscale(GPUFilterContext *gpu, const Frame *input, Frame *output) {
    (void)gpu;
    (void)input;
    (void)output;
    return report_disabled /* module: gpu/gpu_filters */ ();
}

int gpu_blur3x3(GPUFilterContext *gpu, const Frame *input, Frame *output) {
    (void)gpu;
    (void)input;
    (void)output;
    return report_disabled /* module: gpu/gpu_filters */ ();
}

int gpu_blur5x5(GPUFilterContext *gpu, const Frame *input, Frame *output) {
    (void)gpu;
    (void)input;
    (void)output;
    return report_disabled /* module: gpu/gpu_filters */ ();
}

int gpu_blur9x9(GPUFilterContext *gpu, const Frame *input, Frame *output) {
    (void)gpu;
    (void)input;
    (void)output;
    return report_disabled /* module: gpu/gpu_filters */ ();
}

int gpu_blur13x13(GPUFilterContext *gpu, const Frame *input, Frame *output) {
    (void)gpu;
    (void)input;
    (void)output;
    return report_disabled /* module: gpu/gpu_filters */ ();
}

int gpu_grayscale_with_upload_callback(GPUFilterContext *gpu, Frame *input, Frame *output, GPUFrameUploadedCallback callback, void *user_data) {
    (void)gpu;
    (void)input;
    (void)output;
    (void)callback;
    (void)user_data;
    return report_disabled /* module: gpu/gpu_filters */ ();
}

int gpu_blur3x3_with_upload_callback(GPUFilterContext *gpu, Frame *input, Frame *output, GPUFrameUploadedCallback callback, void *user_data) {
    (void)gpu;
    (void)input;
    (void)output;
    (void)callback;
    (void)user_data;
    return report_disabled /* module: gpu/gpu_filters */ ();
}

int gpu_blur5x5_with_upload_callback(GPUFilterContext *gpu, Frame *input, Frame *output, GPUFrameUploadedCallback callback, void *user_data) {
    (void)gpu;
    (void)input;
    (void)output;
    (void)callback;
    (void)user_data;
    return report_disabled /* module: gpu/gpu_filters */ ();
}

int gpu_blur9x9_with_upload_callback(GPUFilterContext *gpu, Frame *input, Frame *output, GPUFrameUploadedCallback callback, void *user_data) {
    (void)gpu;
    (void)input;
    (void)output;
    (void)callback;
    (void)user_data;
    return report_disabled /* module: gpu/gpu_filters */ ();
}

int gpu_blur13x13_with_upload_callback(GPUFilterContext *gpu, Frame *input, Frame *output, GPUFrameUploadedCallback callback, void *user_data) {
    (void)gpu;
    (void)input;
    (void)output;
    (void)callback;
    (void)user_data;
    return report_disabled /* module: gpu/gpu_filters */ ();
}
