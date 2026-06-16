#ifndef VIDEOCOMPUTEPIPELINE_CPU_FILTERS_H
#define VIDEOCOMPUTEPIPELINE_CPU_FILTERS_H

#include "core/frame.h"

int cpu_grayscale(const Frame *input, Frame *output);
int cpu_blur3x3(const Frame *input, Frame *output);
int cpu_blur5x5(const Frame *input, Frame *output);
int cpu_blur9x9(const Frame *input, Frame *output);

#endif
