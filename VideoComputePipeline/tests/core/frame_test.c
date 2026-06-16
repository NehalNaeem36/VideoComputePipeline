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
    frame_init(&frame);

    TEST_ASSERT(frame_alloc(&frame, 4, 3, FRAME_FORMAT_RGB24) == 0);
    TEST_ASSERT(frame_is_valid(&frame));
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
    frame_init(&copy);
    TEST_ASSERT(frame_copy(&copy, &frame) == 0);
    TEST_ASSERT(frame_is_valid(&copy));
    TEST_ASSERT(copy.index == 42);
    TEST_ASSERT(copy.data != frame.data);
    TEST_ASSERT(memcmp(copy.data, frame.data, frame.size) == 0);

    Frame moved;
    frame_init(&moved);
    TEST_ASSERT(frame_move(&moved, &copy) == 0);
    TEST_ASSERT(frame_is_valid(&moved));
    TEST_ASSERT(!frame_is_valid(&copy));
    TEST_ASSERT(copy.data == NULL);

    frame_free(&frame);
    frame_free(&copy);
    frame_free(&moved);
    TEST_ASSERT(frame.data == NULL);
    TEST_ASSERT(moved.data == NULL);
    return 0;
}

static int frame_test_calculate_size(void) {
    TEST_ASSERT(frame_calculate_stride(8, FRAME_FORMAT_RGB24) == 24);
    TEST_ASSERT(frame_calculate_stride(8, FRAME_FORMAT_GRAY8) == 8);
    TEST_ASSERT(frame_calculate_size(8, 2, FRAME_FORMAT_RGB24) == 48);
    TEST_ASSERT(frame_calculate_size(8, 2, FRAME_FORMAT_GRAY8) == 16);
    TEST_ASSERT(frame_calculate_size(0, 2, FRAME_FORMAT_RGB24) == 0);
    TEST_ASSERT(frame_calculate_size(8, 0, FRAME_FORMAT_RGB24) == 0);
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
    return 0;
}
