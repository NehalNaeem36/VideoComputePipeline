#include "pipeline/frame_queue.h"

#include <stdio.h>

#define TEST_ASSERT(expr) do { if (!(expr)) { fprintf(stderr, "fail: %s:%d: %s\n", __FILE__, __LINE__, #expr); return 1; } } while (0)

int main(void) {
    FrameQueue queue;
    Frame in;
    Frame out;
    frame_init /* module: core/frame */ (&in);
    frame_init /* module: core/frame */ (&out);
    TEST_ASSERT(frame_queue_init /* module: pipeline/frame_queue */ (&queue, 2) == 0);
    TEST_ASSERT(frame_alloc /* module: core/frame */ (&in, 1, 1, FRAME_FORMAT_RGB24) == 0);
    TEST_ASSERT(frame_queue_push /* module: pipeline/frame_queue */ (&queue, &in) == 0);
    TEST_ASSERT(frame_queue_size /* module: pipeline/frame_queue */ (&queue) == 1);
    TEST_ASSERT(frame_queue_pop /* module: pipeline/frame_queue */ (&queue, &out) == 0);
    TEST_ASSERT(frame_is_valid /* module: core/frame */ (&out));
    frame_free /* module: core/frame */ (&out);
    frame_queue_free /* module: pipeline/frame_queue */ (&queue);
    return 0;
}
