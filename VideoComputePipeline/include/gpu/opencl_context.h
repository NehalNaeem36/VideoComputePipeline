#ifndef VIDEOCOMPUTEPIPELINE_OPENCL_CONTEXT_H
#define VIDEOCOMPUTEPIPELINE_OPENCL_CONTEXT_H

#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>

typedef struct {
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
} OpenCLContext;

int opencl_context_init(OpenCLContext *ctx);
void opencl_context_print_info(const OpenCLContext *ctx);
void opencl_context_release(OpenCLContext *ctx);

#endif
