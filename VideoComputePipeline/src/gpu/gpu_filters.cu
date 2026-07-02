/*
 * GPU filters module: runs CUDA grayscale and box-blur kernels over RGB24
 * frames. The filter pipeline calls this module for --mode gpu while core
 * Frame ownership remains with frame pools and video modules.
 */
#include "gpu/gpu_filters.h"
#include "utils/logger.h"

#include <cuda_runtime.h>

#include <stdio.h>
#include <string.h>

static const int CUDA_BLOCK_X = 16;
static const int CUDA_BLOCK_Y = 16;

static const char *cuda_error_name(cudaError_t err) {
    return cudaGetErrorName(err);
}

static void log_cuda_error(const char *operation, cudaError_t err) {
    log_error /* module: utils/logger */ ("%s failed: %s (%d)", operation, cuda_error_name(err), (int)err);
}

static cudaStream_t gpu_stream(const GPUFilterContext *gpu) {
    return (cudaStream_t)gpu->stream;
}

static cudaEvent_t gpu_event(void *event_ptr) {
    return (cudaEvent_t)event_ptr;
}

static int create_event(void **event_ptr) {
    cudaEvent_t event = NULL;
    const cudaError_t err = cudaEventCreate(&event);
    if (err != cudaSuccess) {
        log_cuda_error /* module: gpu/gpu_filters */ ("cudaEventCreate", err);
        return -1;
    }
    *event_ptr = (void *)event;
    return 0;
}

static int record_elapsed_ms(cudaStream_t stream, void *start_ptr, void *stop_ptr, double *out_ms) {
    cudaEvent_t start = gpu_event /* module: gpu/gpu_filters */ (start_ptr);
    cudaEvent_t stop = gpu_event /* module: gpu/gpu_filters */ (stop_ptr);
    cudaError_t err = cudaEventRecord(stop, stream);
    if (err != cudaSuccess) {
        log_cuda_error /* module: gpu/gpu_filters */ ("cudaEventRecord", err);
        return -1;
    }
    err = cudaEventSynchronize(stop);
    if (err != cudaSuccess) {
        log_cuda_error /* module: gpu/gpu_filters */ ("cudaEventSynchronize", err);
        return -1;
    }
    float elapsed = 0.0f;
    err = cudaEventElapsedTime(&elapsed, start, stop);
    if (err != cudaSuccess) {
        log_cuda_error /* module: gpu/gpu_filters */ ("cudaEventElapsedTime", err);
        return -1;
    }
    *out_ms = (double)elapsed;
    return 0;
}

__device__ static int clamp_int_device(int value, int min_value, int max_value) {
    if (value < min_value) {
        return min_value;
    }
    if (value > max_value) {
        return max_value;
    }
    return value;
}

__global__ static void grayscale_rgb24_kernel(const unsigned char *input,
                                              unsigned char *output,
                                              int width,
                                              int height,
                                              size_t stride) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }

    const size_t offset = (size_t)y * stride + (size_t)x * 3u;
    const unsigned char r = input[offset + 0u];
    const unsigned char g = input[offset + 1u];
    const unsigned char b = input[offset + 2u];
    const unsigned char gray = (unsigned char)(0.299 * (double)r + 0.587 * (double)g + 0.114 * (double)b + 0.5);
    output[offset + 0u] = gray;
    output[offset + 1u] = gray;
    output[offset + 2u] = gray;
}

__global__ static void box_blur_rgb24_kernel(const unsigned char *input,
                                             unsigned char *output,
                                             int width,
                                             int height,
                                             size_t stride,
                                             int kernel_size) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }

    const int start = -(kernel_size / 2);
    const int end = start + kernel_size;
    const int area = kernel_size * kernel_size;
    int sum_r = 0;
    int sum_g = 0;
    int sum_b = 0;

    for (int ky = start; ky < end; ++ky) {
        const int sy = clamp_int_device /* module: gpu/gpu_filters */ (y + ky, 0, height - 1);
        const unsigned char *src_row = input + (size_t)sy * stride;
        for (int kx = start; kx < end; ++kx) {
            const int sx = clamp_int_device /* module: gpu/gpu_filters */ (x + kx, 0, width - 1);
            const unsigned char *pixel = src_row + (size_t)sx * 3u;
            sum_r += pixel[0];
            sum_g += pixel[1];
            sum_b += pixel[2];
        }
    }

    const size_t offset = (size_t)y * stride + (size_t)x * 3u;
    output[offset + 0u] = (unsigned char)(sum_r / area);
    output[offset + 1u] = (unsigned char)(sum_g / area);
    output[offset + 2u] = (unsigned char)(sum_b / area);
}

extern "C" int gpu_filters_init(GPUFilterContext *gpu) {
    if (!gpu) {
        return -1;
    }

    memset(gpu, 0, sizeof(*gpu));

    int device_id = 0;
    cudaError_t err = cudaGetDevice(&device_id);
    if (err != cudaSuccess) {
        log_cuda_error /* module: gpu/gpu_filters */ ("cudaGetDevice", err);
        return -1;
    }

    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, device_id);
    if (err != cudaSuccess) {
        log_cuda_error /* module: gpu/gpu_filters */ ("cudaGetDeviceProperties", err);
        return -1;
    }

    cudaStream_t stream = NULL;
    err = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    if (err != cudaSuccess) {
        log_cuda_error /* module: gpu/gpu_filters */ ("cudaStreamCreateWithFlags", err);
        return -1;
    }

    gpu->stream = (void *)stream;
    gpu->device_id = device_id;
    strncpy(gpu->device_name, prop.name, sizeof(gpu->device_name) - 1u);
    gpu->device_name[sizeof(gpu->device_name) - 1u] = '\0';

    if (create_event /* module: gpu/gpu_filters */ (&gpu->upload_start_event) != 0 ||
        create_event /* module: gpu/gpu_filters */ (&gpu->upload_stop_event) != 0 ||
        create_event /* module: gpu/gpu_filters */ (&gpu->kernel_start_event) != 0 ||
        create_event /* module: gpu/gpu_filters */ (&gpu->kernel_stop_event) != 0 ||
        create_event /* module: gpu/gpu_filters */ (&gpu->download_start_event) != 0 ||
        create_event /* module: gpu/gpu_filters */ (&gpu->download_stop_event) != 0) {
        gpu_filters_release /* module: gpu/gpu_filters */ (gpu);
        return -1;
    }

    return 0;
}

extern "C" void gpu_filters_print_info(const GPUFilterContext *gpu) {
    if (!gpu) {
        return;
    }
    printf("CUDA filter device: %s\n", gpu->device_name[0] ? gpu->device_name : "unknown");
}

extern "C" void gpu_filters_release(GPUFilterContext *gpu) {
    if (!gpu) {
        return;
    }

    if (gpu->input_buffer) {
        cudaFree(gpu->input_buffer);
    }
    if (gpu->output_buffer) {
        cudaFree(gpu->output_buffer);
    }
    if (gpu->upload_start_event) {
        cudaEventDestroy(gpu_event /* module: gpu/gpu_filters */ (gpu->upload_start_event));
    }
    if (gpu->upload_stop_event) {
        cudaEventDestroy(gpu_event /* module: gpu/gpu_filters */ (gpu->upload_stop_event));
    }
    if (gpu->kernel_start_event) {
        cudaEventDestroy(gpu_event /* module: gpu/gpu_filters */ (gpu->kernel_start_event));
    }
    if (gpu->kernel_stop_event) {
        cudaEventDestroy(gpu_event /* module: gpu/gpu_filters */ (gpu->kernel_stop_event));
    }
    if (gpu->download_start_event) {
        cudaEventDestroy(gpu_event /* module: gpu/gpu_filters */ (gpu->download_start_event));
    }
    if (gpu->download_stop_event) {
        cudaEventDestroy(gpu_event /* module: gpu/gpu_filters */ (gpu->download_stop_event));
    }
    if (gpu->stream) {
        cudaStreamDestroy(gpu_stream /* module: gpu/gpu_filters */ (gpu));
    }

    memset(gpu, 0, sizeof(*gpu));
}

static int ensure_buffers(GPUFilterContext *gpu, size_t size) {
    if (gpu->buffer_size >= size && gpu->input_buffer && gpu->output_buffer) {
        return 0;
    }

    if (gpu->input_buffer) {
        cudaFree(gpu->input_buffer);
        gpu->input_buffer = NULL;
    }
    if (gpu->output_buffer) {
        cudaFree(gpu->output_buffer);
        gpu->output_buffer = NULL;
    }
    gpu->buffer_size = 0;

    cudaError_t err = cudaMalloc(&gpu->input_buffer, size);
    if (err != cudaSuccess) {
        log_cuda_error /* module: gpu/gpu_filters */ ("cudaMalloc input", err);
        return -1;
    }

    err = cudaMalloc(&gpu->output_buffer, size);
    if (err != cudaSuccess) {
        log_cuda_error /* module: gpu/gpu_filters */ ("cudaMalloc output", err);
        cudaFree(gpu->input_buffer);
        gpu->input_buffer = NULL;
        return -1;
    }

    gpu->buffer_size = size;
    return 0;
}

static int ensure_rgb_output(const Frame *input, Frame *output) {
    if (!frame_is_valid /* module: core/frame */ (input) || !output || input->format != FRAME_FORMAT_RGB24) {
        return -1;
    }

    if (!frame_is_valid /* module: core/frame */ (output) ||
        output->width != input->width ||
        output->height != input->height ||
        output->format != FRAME_FORMAT_RGB24) {
        if (frame_alloc /* module: core/frame */ (output, input->width, input->height, FRAME_FORMAT_RGB24) != 0) {
            log_error /* module: utils/logger */ ("failed to allocate CUDA filter output frame: %dx%d, %zu bytes",
                      input->width,
                      input->height,
                      input->size);
            return -1;
        }
    }

    output->index = input->index;
    return 0;
}

static int run_filter(GPUFilterContext *gpu,
                      Frame *input,
                      Frame *output,
                      int kernel_size,
                      GPUFrameUploadedCallback callback,
                      void *user_data) {
    if (!gpu || !input || !output || ensure_rgb_output /* module: gpu/gpu_filters */ (input, output) != 0) {
        return -1;
    }
    if (kernel_size != 0 && kernel_size != 3 && kernel_size != 5 && kernel_size != 9 && kernel_size != 13) {
        return -1;
    }

    const int width = input->width;
    const int height = input->height;
    const size_t stride = input->stride;
    const size_t input_size = input->size;

    if (stride > (size_t)2147483647 || input_size != output->size) {
        log_error /* module: utils/logger */ ("unsupported CUDA filter frame layout");
        return -1;
    }

    if (ensure_buffers /* module: gpu/gpu_filters */ (gpu, input_size) != 0) {
        return -1;
    }

    cudaStream_t stream = gpu_stream /* module: gpu/gpu_filters */ (gpu);
    cudaError_t err = cudaEventRecord(gpu_event /* module: gpu/gpu_filters */ (gpu->upload_start_event), stream);
    if (err != cudaSuccess) {
        log_cuda_error /* module: gpu/gpu_filters */ ("cudaEventRecord upload start", err);
        return -1;
    }
    err = cudaMemcpyAsync(gpu->input_buffer, input->data, input_size, cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) {
        log_cuda_error /* module: gpu/gpu_filters */ ("cudaMemcpyAsync upload", err);
        return -1;
    }
    if (record_elapsed_ms /* module: gpu/gpu_filters */ (stream,
                                                         gpu->upload_start_event,
                                                         gpu->upload_stop_event,
                                                         &gpu->last_upload_ms) != 0) {
        return -1;
    }

    if (callback) {
        callback(user_data, input);
    }

    const dim3 block(CUDA_BLOCK_X, CUDA_BLOCK_Y);
    const dim3 grid((unsigned int)((width + CUDA_BLOCK_X - 1) / CUDA_BLOCK_X),
                    (unsigned int)((height + CUDA_BLOCK_Y - 1) / CUDA_BLOCK_Y));

    err = cudaEventRecord(gpu_event /* module: gpu/gpu_filters */ (gpu->kernel_start_event), stream);
    if (err != cudaSuccess) {
        log_cuda_error /* module: gpu/gpu_filters */ ("cudaEventRecord kernel start", err);
        return -1;
    }

    if (kernel_size == 0) {
        grayscale_rgb24_kernel<<<grid, block, 0, stream>>>((const unsigned char *)gpu->input_buffer,
                                                           (unsigned char *)gpu->output_buffer,
                                                           width,
                                                           height,
                                                           stride);
    } else {
        box_blur_rgb24_kernel<<<grid, block, 0, stream>>>((const unsigned char *)gpu->input_buffer,
                                                          (unsigned char *)gpu->output_buffer,
                                                          width,
                                                          height,
                                                          stride,
                                                          kernel_size);
    }
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        log_cuda_error /* module: gpu/gpu_filters */ ("CUDA filter kernel launch", err);
        return -1;
    }
    if (record_elapsed_ms /* module: gpu/gpu_filters */ (stream,
                                                         gpu->kernel_start_event,
                                                         gpu->kernel_stop_event,
                                                         &gpu->last_kernel_ms) != 0) {
        return -1;
    }

    err = cudaEventRecord(gpu_event /* module: gpu/gpu_filters */ (gpu->download_start_event), stream);
    if (err != cudaSuccess) {
        log_cuda_error /* module: gpu/gpu_filters */ ("cudaEventRecord download start", err);
        return -1;
    }
    err = cudaMemcpyAsync(output->data, gpu->output_buffer, output->size, cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess) {
        log_cuda_error /* module: gpu/gpu_filters */ ("cudaMemcpyAsync download", err);
        return -1;
    }
    if (record_elapsed_ms /* module: gpu/gpu_filters */ (stream,
                                                         gpu->download_start_event,
                                                         gpu->download_stop_event,
                                                         &gpu->last_download_ms) != 0) {
        return -1;
    }

    return 0;
}

extern "C" int gpu_grayscale(GPUFilterContext *gpu, const Frame *input, Frame *output) {
    return run_filter(gpu, (Frame *)input, output, 0, NULL, NULL);
}

extern "C" int gpu_blur3x3(GPUFilterContext *gpu, const Frame *input, Frame *output) {
    return run_filter(gpu, (Frame *)input, output, 3, NULL, NULL);
}

extern "C" int gpu_blur5x5(GPUFilterContext *gpu, const Frame *input, Frame *output) {
    return run_filter(gpu, (Frame *)input, output, 5, NULL, NULL);
}

extern "C" int gpu_blur9x9(GPUFilterContext *gpu, const Frame *input, Frame *output) {
    return run_filter(gpu, (Frame *)input, output, 9, NULL, NULL);
}

extern "C" int gpu_blur13x13(GPUFilterContext *gpu, const Frame *input, Frame *output) {
    return run_filter(gpu, (Frame *)input, output, 13, NULL, NULL);
}

extern "C" int gpu_grayscale_with_upload_callback(GPUFilterContext *gpu, Frame *input, Frame *output, GPUFrameUploadedCallback callback, void *user_data) {
    return run_filter(gpu, input, output, 0, callback, user_data);
}

extern "C" int gpu_blur3x3_with_upload_callback(GPUFilterContext *gpu, Frame *input, Frame *output, GPUFrameUploadedCallback callback, void *user_data) {
    return run_filter(gpu, input, output, 3, callback, user_data);
}

extern "C" int gpu_blur5x5_with_upload_callback(GPUFilterContext *gpu, Frame *input, Frame *output, GPUFrameUploadedCallback callback, void *user_data) {
    return run_filter(gpu, input, output, 5, callback, user_data);
}

extern "C" int gpu_blur9x9_with_upload_callback(GPUFilterContext *gpu, Frame *input, Frame *output, GPUFrameUploadedCallback callback, void *user_data) {
    return run_filter(gpu, input, output, 9, callback, user_data);
}

extern "C" int gpu_blur13x13_with_upload_callback(GPUFilterContext *gpu, Frame *input, Frame *output, GPUFrameUploadedCallback callback, void *user_data) {
    return run_filter(gpu, input, output, 13, callback, user_data);
}
