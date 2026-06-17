#include "gpu/opencl_context.h"

#include <stdio.h>

int main(void) {
    OpenCLContext ctx;
    if (opencl_context_init /* module: gpu/opencl_context */ (&ctx) != 0) {
        printf("opencl_context_test skipped: no OpenCL device available\n");
        return 0;
    }
    opencl_context_print_info /* module: gpu/opencl_context */ (&ctx);
    opencl_context_release /* module: gpu/opencl_context */ (&ctx);
    return 0;
}
