#ifndef VIDEOCOMPUTEPIPELINE_GPU_OPENCL_PROGRAM_H
#define VIDEOCOMPUTEPIPELINE_GPU_OPENCL_PROGRAM_H

#include "gpu/opencl_context.h"

/**
 * OpenCL program wrapper
 */
typedef struct {
    void *program;
    void *kernel_grayscale;
    void *kernel_blur3x3;
    void *kernel_blur5x5;
    void *kernel_blur9x9;
} OpenCLProgram;

/**
 * Create and build OpenCL program from source
 */
OpenCLProgram* opencl_program_create(OpenCLContext *ctx, const char *kernel_path);

/**
 * Free OpenCL program resources
 */
void opencl_program_destroy(OpenCLProgram *prog);

/**
 * Get kernel object by name
 */
void* opencl_program_get_kernel(OpenCLProgram *prog, const char *kernel_name);

#endif // VIDEOCOMPUTEPIPELINE_GPU_OPENCL_PROGRAM_H
