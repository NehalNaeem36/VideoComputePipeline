#include "gpu/opencl_context.h"
#include "utils/logger.h"

#include <stdio.h>
#include <string.h>

static int choose_device(OpenCLContext *ctx, cl_device_type type) {
    cl_uint platform_count = 0;
    if (clGetPlatformIDs(0, NULL, &platform_count) != CL_SUCCESS || platform_count == 0) {
        return -1;
    }

    cl_platform_id platforms[16];
    if (platform_count > 16) {
        platform_count = 16;
    }
    if (clGetPlatformIDs(platform_count, platforms, NULL) != CL_SUCCESS) {
        return -1;
    }

    for (cl_uint i = 0; i < platform_count; ++i) {
        cl_uint device_count = 0;
        if (clGetDeviceIDs(platforms[i], type, 0, NULL, &device_count) != CL_SUCCESS || device_count == 0) {
            continue;
        }

        cl_device_id devices[16];
        if (device_count > 16) {
            device_count = 16;
        }
        if (clGetDeviceIDs(platforms[i], type, device_count, devices, NULL) == CL_SUCCESS) {
            ctx->platform = platforms[i];
            ctx->device = devices[0];
            return 0;
        }
    }

    return -1;
}

int opencl_context_init(OpenCLContext *ctx) {
    if (!ctx) {
        return -1;
    }

    memset(ctx, 0, sizeof(*ctx));
    if (choose_device(ctx, CL_DEVICE_TYPE_GPU) != 0 &&
        choose_device(ctx, CL_DEVICE_TYPE_CPU) != 0) {
        return -1;
    }

    cl_int err = CL_SUCCESS;
    ctx->context = clCreateContext(NULL, 1, &ctx->device, NULL, NULL, &err);
    if (err != CL_SUCCESS || !ctx->context) {
        opencl_context_release(ctx);
        return -1;
    }

    const cl_queue_properties props[] = { 0 };
    ctx->queue = clCreateCommandQueueWithProperties(ctx->context, ctx->device, props, &err);
    if (err != CL_SUCCESS || !ctx->queue) {
        opencl_context_release(ctx);
        return -1;
    }

    return 0;
}

void opencl_context_print_info(const OpenCLContext *ctx) {
    if (!ctx || !ctx->device || !ctx->platform) {
        return;
    }

    char platform_name[256] = {0};
    char device_name[256] = {0};
    clGetPlatformInfo(ctx->platform, CL_PLATFORM_NAME, sizeof(platform_name), platform_name, NULL);
    clGetDeviceInfo(ctx->device, CL_DEVICE_NAME, sizeof(device_name), device_name, NULL);
    printf("OpenCL platform: %s\n", platform_name);
    printf("OpenCL device: %s\n", device_name);
}

void opencl_context_release(OpenCLContext *ctx) {
    if (!ctx) {
        return;
    }

    if (ctx->queue) {
        clReleaseCommandQueue(ctx->queue);
    }
    if (ctx->context) {
        clReleaseContext(ctx->context);
    }
    memset(ctx, 0, sizeof(*ctx));
}
