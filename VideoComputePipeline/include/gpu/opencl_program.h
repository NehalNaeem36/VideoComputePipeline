#ifndef VIDEOCOMPUTEPIPELINE_OPENCL_PROGRAM_H
#define VIDEOCOMPUTEPIPELINE_OPENCL_PROGRAM_H

#include "gpu/opencl_context.h"

typedef struct {
    cl_program program;
} OpenCLProgram;

int opencl_program_build(OpenCLProgram *program, OpenCLContext *ctx, const char *kernel_path);
void opencl_program_release(OpenCLProgram *program);

#endif
