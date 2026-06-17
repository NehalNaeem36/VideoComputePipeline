#include "gpu/opencl_program.h"

#include <stdio.h>

#ifndef VCP_KERNEL_DIR
#define VCP_KERNEL_DIR "kernels"
#endif

int main(void) {
    OpenCLContext ctx;
    OpenCLProgram program;
    if (opencl_context_init /* module: gpu/opencl_context */ (&ctx) != 0) {
        printf("opencl_program_test skipped: no OpenCL device available\n");
        return 0;
    }
    if (opencl_program_build /* module: gpu/opencl_program */ (&program, &ctx, VCP_KERNEL_DIR "/grayscale.cl") != 0) {
        opencl_context_release /* module: gpu/opencl_context */ (&ctx);
        return 1;
    }
    opencl_program_release /* module: gpu/opencl_program */ (&program);
    opencl_context_release /* module: gpu/opencl_context */ (&ctx);
    return 0;
}
