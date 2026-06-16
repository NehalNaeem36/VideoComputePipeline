#ifndef VIDEOCOMPUTEPIPELINE_GPU_FILTERS_H
#define VIDEOCOMPUTEPIPELINE_GPU_FILTERS_H

#include "core/frame.h"
#include "gpu/opencl_context.h"
#include "gpu/opencl_program.h"

typedef struct {
    OpenCLContext ctx;
    OpenCLProgram grayscale_program;
    OpenCLProgram blur3x3_program;
    OpenCLProgram blur5x5_program;
    OpenCLProgram blur9x9_program;
    OpenCLProgram blur13x13_program;
    cl_kernel grayscale_kernel;
    cl_kernel blur3x3_kernel;
    cl_kernel blur5x5_kernel;
    cl_kernel blur9x9_kernel;
    cl_kernel blur13x13_kernel;
    cl_mem input_buffer;
    cl_mem output_buffer;
    size_t buffer_size;
    double last_upload_ms;
    double last_kernel_ms;
    double last_download_ms;
} GPUFilterContext;

typedef void (*GPUFrameUploadedCallback)(void *user_data, Frame *input);

int gpu_filters_init(GPUFilterContext *gpu);
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

#endif
