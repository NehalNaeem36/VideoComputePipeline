#include "cpu/cpu_filters.h"
#include "gpu/gpu_filters.h"

#include <stdio.h>
#include <string.h>

#define TEST_ASSERT(expr) do { if (!(expr)) { fprintf(stderr, "fail: %s:%d: %s\n", __FILE__, __LINE__, #expr); return 1; } } while (0)

typedef int (*FilterFn)(const Frame *input, Frame *output);
typedef int (*GPUFilterFn)(GPUFilterContext *gpu, const Frame *input, Frame *output);

static void fill_test_pattern(Frame *frame) {
    for (int y = 0; y < frame->height; ++y) {
        unsigned char *row = frame->data + (size_t)y * frame->stride;
        for (int x = 0; x < frame->width; ++x) {
            row[x * 3 + 0] = (unsigned char)((x * 17 + y * 3) & 0xff);
            row[x * 3 + 1] = (unsigned char)((x * 5 + y * 29) & 0xff);
            row[x * 3 + 2] = (unsigned char)((x * 11 + y * 7) & 0xff);
        }
    }
}

static int compare_filter(GPUFilterContext *gpu, const Frame *input, FilterFn cpu_fn, GPUFilterFn gpu_fn, const char *name) {
    Frame cpu_output;
    Frame gpu_output;
    frame_init /* module: core/frame */ (&cpu_output);
    frame_init /* module: core/frame */ (&gpu_output);

    if (cpu_fn(input, &cpu_output) != 0 || gpu_fn(gpu, input, &gpu_output) != 0) {
        frame_free /* module: core/frame */ (&cpu_output);
        frame_free /* module: core/frame */ (&gpu_output);
        fprintf(stderr, "filter failed: %s\n", name);
        return -1;
    }

    const int matches = cpu_output.size == gpu_output.size &&
                        memcmp(cpu_output.data, gpu_output.data, cpu_output.size) == 0;
    frame_free /* module: core/frame */ (&cpu_output);
    frame_free /* module: core/frame */ (&gpu_output);
    if (!matches) {
        fprintf(stderr, "CPU/CUDA output mismatch: %s\n", name);
        return -1;
    }
    return 0;
}

int main(void) {
    GPUFilterContext gpu;
    if (gpu_filters_init /* module: gpu/gpu_filters */ (&gpu) != 0) {
        printf("gpu_filters_test skipped: CUDA GPU filters unavailable\n");
        return 0;
    }

    Frame input;
    frame_init /* module: core/frame */ (&input);
    TEST_ASSERT(frame_alloc /* module: core/frame */ (&input, 17, 15, FRAME_FORMAT_RGB24) == 0);
    fill_test_pattern /* module: tests/gpu/gpu_filters_test */ (&input);

    TEST_ASSERT(compare_filter /* module: tests/gpu/gpu_filters_test */ (&gpu, &input, cpu_grayscale, gpu_grayscale, "grayscale") == 0);
    TEST_ASSERT(compare_filter /* module: tests/gpu/gpu_filters_test */ (&gpu, &input, cpu_blur3x3, gpu_blur3x3, "blur3x3") == 0);
    TEST_ASSERT(compare_filter /* module: tests/gpu/gpu_filters_test */ (&gpu, &input, cpu_blur5x5, gpu_blur5x5, "blur5x5") == 0);
    TEST_ASSERT(compare_filter /* module: tests/gpu/gpu_filters_test */ (&gpu, &input, cpu_blur9x9, gpu_blur9x9, "blur9x9") == 0);
    TEST_ASSERT(compare_filter /* module: tests/gpu/gpu_filters_test */ (&gpu, &input, cpu_blur13x13, gpu_blur13x13, "blur13x13") == 0);

    frame_free /* module: core/frame */ (&input);
    gpu_filters_release /* module: gpu/gpu_filters */ (&gpu);
    return 0;
}
