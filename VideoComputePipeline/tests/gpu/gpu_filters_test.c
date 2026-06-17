#include "gpu/gpu_filters.h"

#include <stdio.h>

#define TEST_ASSERT(expr) do { if (!(expr)) { fprintf(stderr, "fail: %s:%d: %s\n", __FILE__, __LINE__, #expr); return 1; } } while (0)

int main(void) {
    GPUFilterContext gpu;
    if (gpu_filters_init /* module: gpu/gpu_filters */ (&gpu) != 0) {
        printf("gpu_filters_test skipped: no OpenCL device available\n");
        return 0;
    }

    Frame input;
    Frame output;
    frame_init /* module: core/frame */ (&input);
    frame_init /* module: core/frame */ (&output);
    TEST_ASSERT(frame_alloc /* module: core/frame */ (&input, 8, 8, FRAME_FORMAT_RGB24) == 0);
    input.data[0] = 255;
    TEST_ASSERT(gpu_grayscale /* module: gpu/gpu_filters */ (&gpu, &input, &output) == 0);
    TEST_ASSERT(frame_is_valid /* module: core/frame */ (&output));
    TEST_ASSERT(gpu_blur13x13 /* module: gpu/gpu_filters */ (&gpu, &input, &output) == 0);
    frame_free /* module: core/frame */ (&input);
    frame_free /* module: core/frame */ (&output);
    gpu_filters_release /* module: gpu/gpu_filters */ (&gpu);
    return 0;
}
