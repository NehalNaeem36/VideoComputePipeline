#include "gpu/gpu_filters.h"

#include <stdio.h>

#define TEST_ASSERT(expr) do { if (!(expr)) { fprintf(stderr, "fail: %s:%d: %s\n", __FILE__, __LINE__, #expr); return 1; } } while (0)

int main(void) {
    GPUFilterContext gpu;
    if (gpu_filters_init(&gpu) != 0) {
        printf("gpu_filters_test skipped: no OpenCL device available\n");
        return 0;
    }

    Frame input;
    Frame output;
    frame_init(&input);
    frame_init(&output);
    TEST_ASSERT(frame_alloc(&input, 8, 8, FRAME_FORMAT_RGB24) == 0);
    input.data[0] = 255;
    TEST_ASSERT(gpu_grayscale(&gpu, &input, &output) == 0);
    TEST_ASSERT(frame_is_valid(&output));
    TEST_ASSERT(gpu_blur13x13(&gpu, &input, &output) == 0);
    frame_free(&input);
    frame_free(&output);
    gpu_filters_release(&gpu);
    return 0;
}
