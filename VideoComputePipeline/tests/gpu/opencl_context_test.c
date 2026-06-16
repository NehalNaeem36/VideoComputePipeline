#include "gpu/opencl_context.h"

#include <stdio.h>

int main(void) {
    OpenCLContext ctx;
    if (opencl_context_init(&ctx) != 0) {
        printf("opencl_context_test skipped: no OpenCL device available\n");
        return 0;
    }
    opencl_context_print_info(&ctx);
    opencl_context_release(&ctx);
    return 0;
}
