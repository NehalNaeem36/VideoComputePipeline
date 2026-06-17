#include "cpu/cpu_filters.h"

#include <stdio.h>

#define TEST_ASSERT(expr) do { if (!(expr)) { fprintf(stderr, "fail: %s:%d: %s\n", __FILE__, __LINE__, #expr); return 1; } } while (0)

int main(void) {
    Frame input;
    Frame output;
    frame_init /* module: core/frame */ (&input);
    frame_init /* module: core/frame */ (&output);

    TEST_ASSERT(frame_alloc /* module: core/frame */ (&input, 2, 1, FRAME_FORMAT_RGB24) == 0);
    input.index = 7;
    input.data[0] = 255; input.data[1] = 0; input.data[2] = 0;
    input.data[3] = 0; input.data[4] = 255; input.data[5] = 0;

    TEST_ASSERT(cpu_grayscale /* module: cpu/cpu_filters */ (&input, &output) == 0);
    TEST_ASSERT(output.index == 7);
    TEST_ASSERT(output.data[0] == output.data[1] && output.data[1] == output.data[2]);
    TEST_ASSERT(cpu_blur3x3 /* module: cpu/cpu_filters */ (&input, &output) == 0);
    TEST_ASSERT(cpu_blur5x5 /* module: cpu/cpu_filters */ (&input, &output) == 0);
    TEST_ASSERT(cpu_blur9x9 /* module: cpu/cpu_filters */ (&input, &output) == 0);
    TEST_ASSERT(cpu_blur13x13 /* module: cpu/cpu_filters */ (&input, &output) == 0);

    frame_free /* module: core/frame */ (&input);
    frame_free /* module: core/frame */ (&output);
    return 0;
}
