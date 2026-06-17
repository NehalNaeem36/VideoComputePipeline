#include "pipeline/frame_pool.h"

#include <stdio.h>

#define TEST_ASSERT(expr) do { if (!(expr)) { fprintf(stderr, "fail: %s:%d: %s\n", __FILE__, __LINE__, #expr); return 1; } } while (0)

int main(void) {
    FramePool pool;
    Frame frame;
    frame_init /* module: core/frame */ (&frame);

    TEST_ASSERT(frame_pool_init /* module: pipeline/frame_pool */ (&pool, 2, 4, 4, FRAME_FORMAT_RGB24) == 0);
    TEST_ASSERT(frame_pool_available /* module: pipeline/frame_pool */ (&pool) == 2);
    TEST_ASSERT(frame_pool_acquire /* module: pipeline/frame_pool */ (&pool, &frame) == 1);
    TEST_ASSERT(frame_is_valid /* module: core/frame */ (&frame));
    TEST_ASSERT(frame_pool_available /* module: pipeline/frame_pool */ (&pool) == 1);
    TEST_ASSERT(frame_pool_release /* module: pipeline/frame_pool */ (&pool, &frame) == 0);
    TEST_ASSERT(!frame_is_valid /* module: core/frame */ (&frame));
    TEST_ASSERT(frame_pool_available /* module: pipeline/frame_pool */ (&pool) == 2);

    frame_pool_free /* module: pipeline/frame_pool */ (&pool);
    return 0;
}
