#include "core/frame.h"

#include <stdio.h>
#include <string.h>

#define TEST_ASSERT(expr) \
    do { \
        if (!(expr)) { \
            fprintf(stderr, "Assertion failed: %s:%d: %s\n", __FILE__, __LINE__, #expr); \
            return 1; \
        } \
    } while (0)

static int frame_test_allocate_copy_move_free(void) {
    Frame frame;
    frame_init /* module: core/frame */ (&frame);

    TEST_ASSERT(frame_alloc /* module: core/frame */ (&frame, 4, 3, FRAME_FORMAT_RGB24) == 0);
    TEST_ASSERT(frame_is_valid /* module: core/frame */ (&frame));
    TEST_ASSERT(frame.width == 4);
    TEST_ASSERT(frame.height == 3);
    TEST_ASSERT(frame.channels == 3);
    TEST_ASSERT(frame.stride == 12);
    TEST_ASSERT(frame.size == 36);

    frame.index = 42;
    frame.data[0] = 10;
    frame.data[1] = 20;
    frame.data[2] = 30;

    Frame copy;
    frame_init /* module: core/frame */ (&copy);
    TEST_ASSERT(frame_copy /* module: core/frame */ (&copy, &frame) == 0);
    TEST_ASSERT(frame_is_valid /* module: core/frame */ (&copy));
    TEST_ASSERT(copy.index == 42);
    TEST_ASSERT(copy.data != frame.data);
    TEST_ASSERT(memcmp(copy.data, frame.data, frame.size) == 0);

    Frame moved;
    frame_init /* module: core/frame */ (&moved);
    TEST_ASSERT(frame_move /* module: core/frame */ (&moved, &copy) == 0);
    TEST_ASSERT(frame_is_valid /* module: core/frame */ (&moved));
    TEST_ASSERT(!frame_is_valid /* module: core/frame */ (&copy));
    TEST_ASSERT(copy.data == NULL);

    frame_free /* module: core/frame */ (&frame);
    frame_free /* module: core/frame */ (&copy);
    frame_free /* module: core/frame */ (&moved);
    TEST_ASSERT(frame.data == NULL);
    TEST_ASSERT(moved.data == NULL);
    return 0;
}

static int frame_test_calculate_size(void) {
    TEST_ASSERT(frame_calculate_stride /* module: core/frame */ (8, FRAME_FORMAT_RGB24) == 24);
    TEST_ASSERT(frame_calculate_stride /* module: core/frame */ (8, FRAME_FORMAT_GRAY8) == 8);
    TEST_ASSERT(frame_calculate_stride /* module: core/frame */ (8, FRAME_FORMAT_NV12) == 8);
    TEST_ASSERT(frame_calculate_size /* module: core/frame */ (8, 2, FRAME_FORMAT_RGB24) == 48);
    TEST_ASSERT(frame_calculate_size /* module: core/frame */ (8, 2, FRAME_FORMAT_GRAY8) == 16);
    TEST_ASSERT(frame_calculate_size /* module: core/frame */ (8, 2, FRAME_FORMAT_NV12) == 24);
    TEST_ASSERT(frame_calculate_size /* module: core/frame */ (7, 2, FRAME_FORMAT_NV12) == 0);
    TEST_ASSERT(frame_calculate_size /* module: core/frame */ (8, 3, FRAME_FORMAT_NV12) == 0);
    TEST_ASSERT(frame_calculate_size /* module: core/frame */ (0, 2, FRAME_FORMAT_RGB24) == 0);
    TEST_ASSERT(frame_calculate_size /* module: core/frame */ (8, 0, FRAME_FORMAT_RGB24) == 0);
    return 0;
}

static int frame_test_nv12_layout(void) {
    Frame frame;
    frame_init /* module: core/frame */ (&frame);

    TEST_ASSERT(frame_alloc /* module: core/frame */ (&frame, 8, 4, FRAME_FORMAT_NV12) == 0);
    TEST_ASSERT(frame_is_valid /* module: core/frame */ (&frame));
    TEST_ASSERT(frame.channels == 1);
    TEST_ASSERT(frame.stride == 8);
    TEST_ASSERT(frame.size == 48);
    TEST_ASSERT(frame.planes[0] == frame.data);
    TEST_ASSERT(frame.planes[1] == frame.data + 32);
    TEST_ASSERT(frame.linesize[0] == 8);
    TEST_ASSERT(frame.linesize[1] == 8);

    frame_free /* module: core/frame */ (&frame);
    return 0;
}

int main(void) {
    printf("Running frame tests...\n");
    if (frame_test_allocate_copy_move_free() != 0) {
        return 1;
    }
    if (frame_test_calculate_size() != 0) {
        return 1;
    }
    if (frame_test_nv12_layout() != 0) {
        return 1;
    }
    return 0;
}
