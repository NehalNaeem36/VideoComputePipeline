#include "gpu/opencl_program.h"
#include <stdlib.h>

// TODO: Include OpenCL headers when available
// #include <CL/cl.h>

OpenCLProgram* opencl_program_create(OpenCLContext *ctx, const char *kernel_path) {
    // TODO: Implement program compilation and kernel creation
    return NULL;
}

void opencl_program_destroy(OpenCLProgram *prog) {
    // TODO: Implement cleanup
}

void* opencl_program_get_kernel(OpenCLProgram *prog, const char *kernel_name) {
    // TODO: Implement kernel lookup
    return NULL;
}
