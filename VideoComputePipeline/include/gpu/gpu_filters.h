#ifndef VIDEOCOMPUTEPIPELINE_GPU_GPU_FILTERS_H
#define VIDEOCOMPUTEPIPELINE_GPU_GPU_FILTERS_H

#include "core/frame.h"
#include "gpu/opencl_context.h"
#include "gpu/opencl_program.h"

/**
 * Apply grayscale filter to frame (GPU implementation)
 */
void gpu_grayscale(OpenCLContext *ctx, OpenCLProgram *prog, Frame *input, Frame *output);

/**
 * Apply 3x3 box blur filter to frame (GPU implementation)
 */
void gpu_blur_3x3(OpenCLContext *ctx, OpenCLProgram *prog, Frame *input, Frame *output);

/**
 * Apply 5x5 box blur filter to frame (GPU implementation)
 */
void gpu_blur_5x5(OpenCLContext *ctx, OpenCLProgram *prog, Frame *input, Frame *output);

/**
 * Apply 9x9 box blur filter to frame (GPU implementation)
 */
void gpu_blur_9x9(OpenCLContext *ctx, OpenCLProgram *prog, Frame *input, Frame *output);

#endif // VIDEOCOMPUTEPIPELINE_GPU_GPU_FILTERS_H
