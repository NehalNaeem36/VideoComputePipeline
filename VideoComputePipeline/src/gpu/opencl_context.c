#include "gpu/opencl_context.h"
#include <stdlib.h>

// TODO: Include OpenCL headers when available
// #include <CL/cl.h>

OpenCLContext* opencl_context_create(void) {
    // TODO: Implement OpenCL platform/device/context initialization
    return NULL;
}

void opencl_context_destroy(OpenCLContext *ctx) {
    // TODO: Implement OpenCL cleanup
}

const char* opencl_context_get_device_info(OpenCLContext *ctx) {
    // TODO: Implement device info retrieval
    return NULL;
}
