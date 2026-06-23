/*
 * GPU filters module: runs OpenCL grayscale and blur kernels over RGB24 frames.
 * It owns GPU filter execution while reusing OpenCL context/program helpers and
 * returns results to pipeline-managed Frame buffers.
 */
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

static const char *opencl_error_name(cl_int err) {
    switch (err) {
        case CL_SUCCESS:
            return "CL_SUCCESS";
        case CL_DEVICE_NOT_FOUND:
            return "CL_DEVICE_NOT_FOUND";
        case CL_DEVICE_NOT_AVAILABLE:
            return "CL_DEVICE_NOT_AVAILABLE";
        case CL_COMPILER_NOT_AVAILABLE:
            return "CL_COMPILER_NOT_AVAILABLE";
        case CL_MEM_OBJECT_ALLOCATION_FAILURE:
            return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
        case CL_OUT_OF_RESOURCES:
            return "CL_OUT_OF_RESOURCES";
        case CL_OUT_OF_HOST_MEMORY:
            return "CL_OUT_OF_HOST_MEMORY";
        case CL_PROFILING_INFO_NOT_AVAILABLE:
            return "CL_PROFILING_INFO_NOT_AVAILABLE";
        case CL_MEM_COPY_OVERLAP:
            return "CL_MEM_COPY_OVERLAP";
        case CL_IMAGE_FORMAT_MISMATCH:
            return "CL_IMAGE_FORMAT_MISMATCH";
        case CL_IMAGE_FORMAT_NOT_SUPPORTED:
            return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
        case CL_BUILD_PROGRAM_FAILURE:
            return "CL_BUILD_PROGRAM_FAILURE";
        case CL_MAP_FAILURE:
            return "CL_MAP_FAILURE";
        case CL_INVALID_VALUE:
            return "CL_INVALID_VALUE";
        case CL_INVALID_DEVICE_TYPE:
            return "CL_INVALID_DEVICE_TYPE";
        case CL_INVALID_PLATFORM:
            return "CL_INVALID_PLATFORM";
        case CL_INVALID_DEVICE:
            return "CL_INVALID_DEVICE";
        case CL_INVALID_CONTEXT:
            return "CL_INVALID_CONTEXT";
        case CL_INVALID_QUEUE_PROPERTIES:
            return "CL_INVALID_QUEUE_PROPERTIES";
        case CL_INVALID_COMMAND_QUEUE:
            return "CL_INVALID_COMMAND_QUEUE";
        case CL_INVALID_HOST_PTR:
            return "CL_INVALID_HOST_PTR";
        case CL_INVALID_MEM_OBJECT:
            return "CL_INVALID_MEM_OBJECT";
        case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:
            return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
        case CL_INVALID_IMAGE_SIZE:
            return "CL_INVALID_IMAGE_SIZE";
        case CL_INVALID_SAMPLER:
            return "CL_INVALID_SAMPLER";
        case CL_INVALID_BINARY:
            return "CL_INVALID_BINARY";
        case CL_INVALID_BUILD_OPTIONS:
            return "CL_INVALID_BUILD_OPTIONS";
        case CL_INVALID_PROGRAM:
            return "CL_INVALID_PROGRAM";
        case CL_INVALID_PROGRAM_EXECUTABLE:
            return "CL_INVALID_PROGRAM_EXECUTABLE";
        case CL_INVALID_KERNEL_NAME:
            return "CL_INVALID_KERNEL_NAME";
        case CL_INVALID_KERNEL_DEFINITION:
            return "CL_INVALID_KERNEL_DEFINITION";
        case CL_INVALID_KERNEL:
            return "CL_INVALID_KERNEL";
        case CL_INVALID_ARG_INDEX:
            return "CL_INVALID_ARG_INDEX";
        case CL_INVALID_ARG_VALUE:
            return "CL_INVALID_ARG_VALUE";
        case CL_INVALID_ARG_SIZE:
            return "CL_INVALID_ARG_SIZE";
        case CL_INVALID_KERNEL_ARGS:
            return "CL_INVALID_KERNEL_ARGS";
        case CL_INVALID_WORK_DIMENSION:
            return "CL_INVALID_WORK_DIMENSION";
        case CL_INVALID_WORK_GROUP_SIZE:
            return "CL_INVALID_WORK_GROUP_SIZE";
        case CL_INVALID_WORK_ITEM_SIZE:
            return "CL_INVALID_WORK_ITEM_SIZE";
        case CL_INVALID_GLOBAL_OFFSET:
            return "CL_INVALID_GLOBAL_OFFSET";
        case CL_INVALID_EVENT_WAIT_LIST:
            return "CL_INVALID_EVENT_WAIT_LIST";
        case CL_INVALID_EVENT:
            return "CL_INVALID_EVENT";
        case CL_INVALID_OPERATION:
            return "CL_INVALID_OPERATION";
        case CL_INVALID_GL_OBJECT:
            return "CL_INVALID_GL_OBJECT";
        case CL_INVALID_BUFFER_SIZE:
            return "CL_INVALID_BUFFER_SIZE";
        case CL_INVALID_MIP_LEVEL:
            return "CL_INVALID_MIP_LEVEL";
        case CL_INVALID_GLOBAL_WORK_SIZE:
            return "CL_INVALID_GLOBAL_WORK_SIZE";
        default:
            return "CL_UNKNOWN_ERROR";
    }
}

static void log_opencl_error(const char *operation, cl_int err) {
    log_error /* module: utils/logger */ ("%s failed: %s (%d)", operation, opencl_error_name(err), (int)err);
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
    if (opencl_context_init /* module: gpu/opencl_context */ (&gpu->ctx) != 0) {
        return -1;
    }

    char grayscale_path[1024];
    char blur3x3_path[1024];
    char blur5x5_path[1024];
    char blur9x9_path[1024];
    char blur13x13_path[1024];
    if (build_kernel_path /* module: gpu/gpu_filters */ (grayscale_path, sizeof(grayscale_path), "grayscale.cl") != 0 ||
        build_kernel_path /* module: gpu/gpu_filters */ (blur3x3_path, sizeof(blur3x3_path), "blur3x3.cl") != 0 ||
        build_kernel_path /* module: gpu/gpu_filters */ (blur5x5_path, sizeof(blur5x5_path), "blur5x5.cl") != 0 ||
        build_kernel_path /* module: gpu/gpu_filters */ (blur9x9_path, sizeof(blur9x9_path), "blur9x9.cl") != 0 ||
        build_kernel_path /* module: gpu/gpu_filters */ (blur13x13_path, sizeof(blur13x13_path), "blur13x13.cl") != 0) {
        gpu_filters_release /* module: gpu/gpu_filters */ (gpu);
        return -1;
    }

    if (opencl_program_build /* module: gpu/opencl_program */ (&gpu->grayscale_program, &gpu->ctx, grayscale_path) != 0 ||
        opencl_program_build /* module: gpu/opencl_program */ (&gpu->blur3x3_program, &gpu->ctx, blur3x3_path) != 0 ||
        opencl_program_build /* module: gpu/opencl_program */ (&gpu->blur5x5_program, &gpu->ctx, blur5x5_path) != 0 ||
        opencl_program_build /* module: gpu/opencl_program */ (&gpu->blur9x9_program, &gpu->ctx, blur9x9_path) != 0 ||
        opencl_program_build /* module: gpu/opencl_program */ (&gpu->blur13x13_program, &gpu->ctx, blur13x13_path) != 0 ||
        create_kernel /* module: gpu/gpu_filters */ (&gpu->grayscale_program, "grayscale_rgb24", &gpu->grayscale_kernel) != 0 ||
        create_kernel /* module: gpu/gpu_filters */ (&gpu->blur3x3_program, "blur3x3_rgb24", &gpu->blur3x3_kernel) != 0 ||
        create_kernel /* module: gpu/gpu_filters */ (&gpu->blur5x5_program, "blur5x5_rgb24", &gpu->blur5x5_kernel) != 0 ||
        create_kernel /* module: gpu/gpu_filters */ (&gpu->blur9x9_program, "blur9x9_rgb24", &gpu->blur9x9_kernel) != 0 ||
        create_kernel /* module: gpu/gpu_filters */ (&gpu->blur13x13_program, "blur13x13_rgb24", &gpu->blur13x13_kernel) != 0) {
        gpu_filters_release /* module: gpu/gpu_filters */ (gpu);
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
    if (gpu->blur13x13_kernel) {
        clReleaseKernel(gpu->blur13x13_kernel);
    }
    opencl_program_release /* module: gpu/opencl_program */ (&gpu->grayscale_program);
    opencl_program_release /* module: gpu/opencl_program */ (&gpu->blur3x3_program);
    opencl_program_release /* module: gpu/opencl_program */ (&gpu->blur5x5_program);
    opencl_program_release /* module: gpu/opencl_program */ (&gpu->blur9x9_program);
    opencl_program_release /* module: gpu/opencl_program */ (&gpu->blur13x13_program);
    opencl_context_release /* module: gpu/opencl_context */ (&gpu->ctx);
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
        log_opencl_error /* module: gpu/gpu_filters */ ("clCreateBuffer input", err);
        return -1;
    }
    gpu->output_buffer = clCreateBuffer(gpu->ctx.context, CL_MEM_WRITE_ONLY, size, NULL, &err);
    if (err != CL_SUCCESS) {
        log_opencl_error /* module: gpu/gpu_filters */ ("clCreateBuffer output", err);
        return -1;
    }
    gpu->buffer_size = size;
    return 0;
}

static int run_kernel(GPUFilterContext *gpu, cl_kernel kernel, Frame *input, Frame *output, GPUFrameUploadedCallback callback, void *user_data) {
    if (!gpu || !kernel || !frame_is_valid /* module: core/frame */ (input) || input->format != FRAME_FORMAT_RGB24 || !output) {
        return -1;
    }

    const int input_width = input->width;
    const int input_height = input->height;
    const int input_index = input->index;
    const size_t input_stride = input->stride;
    const size_t input_size = input->size;

    if (!frame_is_valid /* module: core/frame */ (output) ||
        output->width != input_width ||
        output->height != input_height ||
        output->format != FRAME_FORMAT_RGB24) {
        if (frame_alloc /* module: core/frame */ (output, input_width, input_height, FRAME_FORMAT_RGB24) != 0) {
            log_error /* module: utils/logger */ ("failed to allocate GPU filter output frame: %dx%d, %zu bytes",
                      input_width,
                      input_height,
                      input_size);
            return -1;
        }
    }
    output->index = input_index;

    if (ensure_buffers /* module: gpu/gpu_filters */ (gpu, input_size) != 0) {
        return -1;
    }

    Timer timer;
    timer_start /* module: benchmark/timer */ (&timer);
    cl_int err = clEnqueueWriteBuffer(gpu->ctx.queue, gpu->input_buffer, CL_TRUE, 0, input_size, input->data, 0, NULL, NULL);
    gpu->last_upload_ms = timer_stop_ms /* module: benchmark/timer */ (&timer);
    if (err != CL_SUCCESS) {
        log_opencl_error /* module: gpu/gpu_filters */ ("clEnqueueWriteBuffer", err);
        return -1;
    }
    if (callback) {
        callback(user_data, input);
    }

    const cl_uint width = (cl_uint)input_width;
    const cl_uint height = (cl_uint)input_height;
    const cl_uint stride = (cl_uint)input_stride;

    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &gpu->input_buffer);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &gpu->output_buffer);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_uint), &width);
    err |= clSetKernelArg(kernel, 3, sizeof(cl_uint), &height);
    err |= clSetKernelArg(kernel, 4, sizeof(cl_uint), &stride);
    if (err != CL_SUCCESS) {
        log_opencl_error /* module: gpu/gpu_filters */ ("clSetKernelArg", err);
        return -1;
    }

    const size_t global[2] = { (size_t)input_width, (size_t)input_height };
    timer_start /* module: benchmark/timer */ (&timer);
    err = clEnqueueNDRangeKernel(gpu->ctx.queue, kernel, 2, NULL, global, NULL, 0, NULL, NULL);
    if (err == CL_SUCCESS) {
        err = clFinish(gpu->ctx.queue);
    }
    gpu->last_kernel_ms = timer_stop_ms /* module: benchmark/timer */ (&timer);
    if (err != CL_SUCCESS) {
        log_opencl_error /* module: gpu/gpu_filters */ ("clEnqueueNDRangeKernel/clFinish", err);
        return -1;
    }

    timer_start /* module: benchmark/timer */ (&timer);
    err = clEnqueueReadBuffer(gpu->ctx.queue, gpu->output_buffer, CL_TRUE, 0, output->size, output->data, 0, NULL, NULL);
    gpu->last_download_ms = timer_stop_ms /* module: benchmark/timer */ (&timer);
    if (err != CL_SUCCESS) {
        log_opencl_error /* module: gpu/gpu_filters */ ("clEnqueueReadBuffer", err);
        return -1;
    }
    return 0;
}

int gpu_grayscale(GPUFilterContext *gpu, const Frame *input, Frame *output) {
    return run_kernel(gpu, gpu ? gpu->grayscale_kernel : NULL, (Frame *)input, output, NULL, NULL);
}

int gpu_blur3x3(GPUFilterContext *gpu, const Frame *input, Frame *output) {
    return run_kernel(gpu, gpu ? gpu->blur3x3_kernel : NULL, (Frame *)input, output, NULL, NULL);
}

int gpu_blur5x5(GPUFilterContext *gpu, const Frame *input, Frame *output) {
    return run_kernel(gpu, gpu ? gpu->blur5x5_kernel : NULL, (Frame *)input, output, NULL, NULL);
}

int gpu_blur9x9(GPUFilterContext *gpu, const Frame *input, Frame *output) {
    return run_kernel(gpu, gpu ? gpu->blur9x9_kernel : NULL, (Frame *)input, output, NULL, NULL);
}

int gpu_blur13x13(GPUFilterContext *gpu, const Frame *input, Frame *output) {
    return run_kernel(gpu, gpu ? gpu->blur13x13_kernel : NULL, (Frame *)input, output, NULL, NULL);
}

int gpu_grayscale_with_upload_callback(GPUFilterContext *gpu, Frame *input, Frame *output, GPUFrameUploadedCallback callback, void *user_data) {
    return run_kernel(gpu, gpu ? gpu->grayscale_kernel : NULL, input, output, callback, user_data);
}

int gpu_blur3x3_with_upload_callback(GPUFilterContext *gpu, Frame *input, Frame *output, GPUFrameUploadedCallback callback, void *user_data) {
    return run_kernel(gpu, gpu ? gpu->blur3x3_kernel : NULL, input, output, callback, user_data);
}

int gpu_blur5x5_with_upload_callback(GPUFilterContext *gpu, Frame *input, Frame *output, GPUFrameUploadedCallback callback, void *user_data) {
    return run_kernel(gpu, gpu ? gpu->blur5x5_kernel : NULL, input, output, callback, user_data);
}

int gpu_blur9x9_with_upload_callback(GPUFilterContext *gpu, Frame *input, Frame *output, GPUFrameUploadedCallback callback, void *user_data) {
    return run_kernel(gpu, gpu ? gpu->blur9x9_kernel : NULL, input, output, callback, user_data);
}

int gpu_blur13x13_with_upload_callback(GPUFilterContext *gpu, Frame *input, Frame *output, GPUFrameUploadedCallback callback, void *user_data) {
    return run_kernel(gpu, gpu ? gpu->blur13x13_kernel : NULL, input, output, callback, user_data);
}
