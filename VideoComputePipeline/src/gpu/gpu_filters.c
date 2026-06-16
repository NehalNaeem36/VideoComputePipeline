#include "gpu/gpu_filters.h"
#include "benchmark/timer.h"
#include "utils/logger.h"

#include <stdio.h>
#include <string.h>

#ifndef VCP_KERNEL_DIR
#define VCP_KERNEL_DIR "kernels"
#endif

static int build_kernel_path(char *dest, size_t dest_size, const char *filename) {
    const int written = snprintf(dest, dest_size, "%s/%s", VCP_KERNEL_DIR, filename);
    return written < 0 || (size_t)written >= dest_size ? -1 : 0;
}

static int create_kernel(OpenCLProgram *program, const char *name, cl_kernel *kernel) {
    cl_int err = CL_SUCCESS;
    *kernel = clCreateKernel(program->program, name, &err);
    return err == CL_SUCCESS && *kernel ? 0 : -1;
}

int gpu_filters_init(GPUFilterContext *gpu) {
    if (!gpu) {
        return -1;
    }

    memset(gpu, 0, sizeof(*gpu));
    if (opencl_context_init(&gpu->ctx) != 0) {
        return -1;
    }

    char grayscale_path[1024];
    char blur3x3_path[1024];
    char blur5x5_path[1024];
    char blur9x9_path[1024];
    if (build_kernel_path(grayscale_path, sizeof(grayscale_path), "grayscale.cl") != 0 ||
        build_kernel_path(blur3x3_path, sizeof(blur3x3_path), "blur3x3.cl") != 0 ||
        build_kernel_path(blur5x5_path, sizeof(blur5x5_path), "blur5x5.cl") != 0 ||
        build_kernel_path(blur9x9_path, sizeof(blur9x9_path), "blur9x9.cl") != 0) {
        gpu_filters_release(gpu);
        return -1;
    }

    if (opencl_program_build(&gpu->grayscale_program, &gpu->ctx, grayscale_path) != 0 ||
        opencl_program_build(&gpu->blur3x3_program, &gpu->ctx, blur3x3_path) != 0 ||
        opencl_program_build(&gpu->blur5x5_program, &gpu->ctx, blur5x5_path) != 0 ||
        opencl_program_build(&gpu->blur9x9_program, &gpu->ctx, blur9x9_path) != 0 ||
        create_kernel(&gpu->grayscale_program, "grayscale_rgb24", &gpu->grayscale_kernel) != 0 ||
        create_kernel(&gpu->blur3x3_program, "blur3x3_rgb24", &gpu->blur3x3_kernel) != 0 ||
        create_kernel(&gpu->blur5x5_program, "blur5x5_rgb24", &gpu->blur5x5_kernel) != 0 ||
        create_kernel(&gpu->blur9x9_program, "blur9x9_rgb24", &gpu->blur9x9_kernel) != 0) {
        gpu_filters_release(gpu);
        return -1;
    }

    return 0;
}

void gpu_filters_release(GPUFilterContext *gpu) {
    if (!gpu) {
        return;
    }

    if (gpu->input_buffer) {
        clReleaseMemObject(gpu->input_buffer);
    }
    if (gpu->output_buffer) {
        clReleaseMemObject(gpu->output_buffer);
    }
    if (gpu->grayscale_kernel) {
        clReleaseKernel(gpu->grayscale_kernel);
    }
    if (gpu->blur3x3_kernel) {
        clReleaseKernel(gpu->blur3x3_kernel);
    }
    if (gpu->blur5x5_kernel) {
        clReleaseKernel(gpu->blur5x5_kernel);
    }
    if (gpu->blur9x9_kernel) {
        clReleaseKernel(gpu->blur9x9_kernel);
    }
    opencl_program_release(&gpu->grayscale_program);
    opencl_program_release(&gpu->blur3x3_program);
    opencl_program_release(&gpu->blur5x5_program);
    opencl_program_release(&gpu->blur9x9_program);
    opencl_context_release(&gpu->ctx);
    memset(gpu, 0, sizeof(*gpu));
}

static int ensure_buffers(GPUFilterContext *gpu, size_t size) {
    if (gpu->buffer_size >= size && gpu->input_buffer && gpu->output_buffer) {
        return 0;
    }

    if (gpu->input_buffer) {
        clReleaseMemObject(gpu->input_buffer);
    }
    if (gpu->output_buffer) {
        clReleaseMemObject(gpu->output_buffer);
    }
    gpu->input_buffer = NULL;
    gpu->output_buffer = NULL;
    gpu->buffer_size = 0;

    cl_int err = CL_SUCCESS;
    gpu->input_buffer = clCreateBuffer(gpu->ctx.context, CL_MEM_READ_ONLY, size, NULL, &err);
    if (err != CL_SUCCESS) {
        return -1;
    }
    gpu->output_buffer = clCreateBuffer(gpu->ctx.context, CL_MEM_WRITE_ONLY, size, NULL, &err);
    if (err != CL_SUCCESS) {
        return -1;
    }
    gpu->buffer_size = size;
    return 0;
}

static int run_kernel(GPUFilterContext *gpu, cl_kernel kernel, const Frame *input, Frame *output) {
    if (!gpu || !kernel || !frame_is_valid(input) || input->format != FRAME_FORMAT_RGB24 || !output) {
        return -1;
    }

    if (!frame_is_valid(output) ||
        output->width != input->width ||
        output->height != input->height ||
        output->format != FRAME_FORMAT_RGB24) {
        if (frame_alloc(output, input->width, input->height, FRAME_FORMAT_RGB24) != 0) {
            return -1;
        }
    }
    output->index = input->index;

    if (ensure_buffers(gpu, input->size) != 0) {
        return -1;
    }

    Timer timer;
    timer_start(&timer);
    cl_int err = clEnqueueWriteBuffer(gpu->ctx.queue, gpu->input_buffer, CL_TRUE, 0, input->size, input->data, 0, NULL, NULL);
    gpu->last_upload_ms = timer_stop_ms(&timer);
    if (err != CL_SUCCESS) {
        return -1;
    }

    const cl_uint width = (cl_uint)input->width;
    const cl_uint height = (cl_uint)input->height;
    const cl_uint stride = (cl_uint)input->stride;

    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &gpu->input_buffer);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &gpu->output_buffer);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_uint), &width);
    err |= clSetKernelArg(kernel, 3, sizeof(cl_uint), &height);
    err |= clSetKernelArg(kernel, 4, sizeof(cl_uint), &stride);
    if (err != CL_SUCCESS) {
        return -1;
    }

    const size_t global[2] = { (size_t)input->width, (size_t)input->height };
    timer_start(&timer);
    err = clEnqueueNDRangeKernel(gpu->ctx.queue, kernel, 2, NULL, global, NULL, 0, NULL, NULL);
    if (err == CL_SUCCESS) {
        err = clFinish(gpu->ctx.queue);
    }
    gpu->last_kernel_ms = timer_stop_ms(&timer);
    if (err != CL_SUCCESS) {
        return -1;
    }

    timer_start(&timer);
    err = clEnqueueReadBuffer(gpu->ctx.queue, gpu->output_buffer, CL_TRUE, 0, output->size, output->data, 0, NULL, NULL);
    gpu->last_download_ms = timer_stop_ms(&timer);
    return err == CL_SUCCESS ? 0 : -1;
}

int gpu_grayscale(GPUFilterContext *gpu, const Frame *input, Frame *output) {
    return run_kernel(gpu, gpu ? gpu->grayscale_kernel : NULL, input, output);
}

int gpu_blur3x3(GPUFilterContext *gpu, const Frame *input, Frame *output) {
    return run_kernel(gpu, gpu ? gpu->blur3x3_kernel : NULL, input, output);
}

int gpu_blur5x5(GPUFilterContext *gpu, const Frame *input, Frame *output) {
    return run_kernel(gpu, gpu ? gpu->blur5x5_kernel : NULL, input, output);
}

int gpu_blur9x9(GPUFilterContext *gpu, const Frame *input, Frame *output) {
    return run_kernel(gpu, gpu ? gpu->blur9x9_kernel : NULL, input, output);
}
