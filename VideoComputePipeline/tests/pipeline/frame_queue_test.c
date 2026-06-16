#include "pipeline/frame_queue.h"

#include <stdio.h>

#define TEST_ASSERT(expr) do { if (!(expr)) { fprintf(stderr, "fail: %s:%d: %s\n", __FILE__, __LINE__, #expr); return 1; } } while (0)

int main(void) {
    FrameQueue queue;
    Frame in;
    Frame out;
    frame_init(&in);
    frame_init(&out);
    TEST_ASSERT(frame_queue_init(&queue, 2) == 0);
    TEST_ASSERT(frame_alloc(&in, 1, 1, FRAME_FORMAT_RGB24) == 0);
    TEST_ASSERT(frame_queue_push(&queue, &in) == 0);
    TEST_ASSERT(frame_queue_size(&queue) == 1);
    TEST_ASSERT(frame_queue_pop(&queue, &out) == 0);
    TEST_ASSERT(frame_is_valid(&out));
    frame_free(&out);
    frame_queue_free(&queue);
    return 0;
}
