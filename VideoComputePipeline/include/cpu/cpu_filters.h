#ifndef VIDEOCOMPUTEPIPELINE_CPU_CPU_FILTERS_H
#define VIDEOCOMPUTEPIPELINE_CPU_CPU_FILTERS_H

#include "core/frame.h"

/**
 * Apply grayscale filter to frame (CPU implementation)
 */
void cpu_grayscale(Frame *input, Frame *output);

/**
 * Apply 3x3 box blur filter to frame (CPU implementation)
 */
void cpu_blur_3x3(Frame *input, Frame *output);

/**
 * Apply 5x5 box blur filter to frame (CPU implementation)
 */
void cpu_blur_5x5(Frame *input, Frame *output);

/**
 * Apply 9x9 box blur filter to frame (CPU implementation)
 */
void cpu_blur_9x9(Frame *input, Frame *output);

#endif // VIDEOCOMPUTEPIPELINE_CPU_CPU_FILTERS_H
