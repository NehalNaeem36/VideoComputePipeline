#include "gpu/opencl_program.h"

#include <stdio.h>

#ifndef VCP_KERNEL_DIR
#define VCP_KERNEL_DIR "kernels"
#endif

int main(void) {
    OpenCLContext ctx;
    OpenCLProgram program;
    if (opencl_context_init(&ctx) != 0) {
        printf("opencl_program_test skipped: no OpenCL device available\n");
        return 0;
    }
    if (opencl_program_build(&program, &ctx, VCP_KERNEL_DIR "/grayscale.cl") != 0) {
        opencl_context_release(&ctx);
        return 1;
    }
    opencl_program_release(&program);
    opencl_context_release(&ctx);
    return 0;
}
