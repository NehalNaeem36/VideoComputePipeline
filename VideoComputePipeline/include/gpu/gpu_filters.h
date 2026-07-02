#ifndef VIDEOCOMPUTEPIPELINE_GPU_FILTERS_H
#define VIDEOCOMPUTEPIPELINE_GPU_FILTERS_H

#include "core/frame.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    void *stream;
    void *upload_start_event;
    void *upload_stop_event;
    void *kernel_start_event;
    void *kernel_stop_event;
    void *download_start_event;
    void *download_stop_event;
    void *input_buffer;
    void *output_buffer;
    size_t buffer_size;
    int device_id;
    char device_name[128];
    double last_upload_ms;
    double last_kernel_ms;
    double last_download_ms;
} GPUFilterContext;

typedef void (*GPUFrameUploadedCallback)(void *user_data, Frame *input);

int gpu_filters_init(GPUFilterContext *gpu);
void gpu_filters_print_info(const GPUFilterContext *gpu);
void gpu_filters_release(GPUFilterContext *gpu);
int gpu_grayscale(GPUFilterContext *gpu, const Frame *input, Frame *output);
int gpu_blur3x3(GPUFilterContext *gpu, const Frame *input, Frame *output);
int gpu_blur5x5(GPUFilterContext *gpu, const Frame *input, Frame *output);
int gpu_blur9x9(GPUFilterContext *gpu, const Frame *input, Frame *output);
int gpu_blur13x13(GPUFilterContext *gpu, const Frame *input, Frame *output);
int gpu_grayscale_with_upload_callback(GPUFilterContext *gpu, Frame *input, Frame *output, GPUFrameUploadedCallback callback, void *user_data);
int gpu_blur3x3_with_upload_callback(GPUFilterContext *gpu, Frame *input, Frame *output, GPUFrameUploadedCallback callback, void *user_data);
int gpu_blur5x5_with_upload_callback(GPUFilterContext *gpu, Frame *input, Frame *output, GPUFrameUploadedCallback callback, void *user_data);
int gpu_blur9x9_with_upload_callback(GPUFilterContext *gpu, Frame *input, Frame *output, GPUFrameUploadedCallback callback, void *user_data);
int gpu_blur13x13_with_upload_callback(GPUFilterContext *gpu, Frame *input, Frame *output, GPUFrameUploadedCallback callback, void *user_data);

#ifdef __cplusplus
}
#endif

#endif
