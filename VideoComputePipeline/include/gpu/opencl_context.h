#ifndef VIDEOCOMPUTEPIPELINE_GPU_OPENCL_CONTEXT_H
#define VIDEOCOMPUTEPIPELINE_GPU_OPENCL_CONTEXT_H

#include <stdint.h>

/**
 * OpenCL context wrapper
 */
typedef struct {
    void *platform_id;
    void *device_id;
    void *context;
    void *command_queue;
} OpenCLContext;

/**
 * Initialize OpenCL context and device
 */
OpenCLContext* opencl_context_create(void);

/**
 * Free OpenCL resources
 */
void opencl_context_destroy(OpenCLContext *ctx);

/**
 * Get device info (name, capabilities)
 */
const char* opencl_context_get_device_info(OpenCLContext *ctx);

#endif // VIDEOCOMPUTEPIPELINE_GPU_OPENCL_CONTEXT_H
