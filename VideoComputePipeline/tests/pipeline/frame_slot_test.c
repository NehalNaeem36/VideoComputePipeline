#include "pipeline/frame_slot.h"

#include <stdio.h>

#define TEST_ASSERT(expr) do { if (!(expr)) { fprintf(stderr, "fail: %s:%d: %s\n", __FILE__, __LINE__, #expr); return 1; } } while (0)

int main(void) {
    FrameSlot slot;
    Frame in;
    Frame out;
    frame_slot_init(&slot);
    frame_init(&in);
    frame_init(&out);
    TEST_ASSERT(frame_alloc(&in, 1, 1, FRAME_FORMAT_RGB24) == 0);
    TEST_ASSERT(frame_slot_put(&slot, &in) == 0);
    TEST_ASSERT(!frame_is_valid(&in));
    TEST_ASSERT(frame_slot_take(&slot, &out) == 0);
    TEST_ASSERT(frame_is_valid(&out));
    frame_free(&out);
    frame_slot_free(&slot);
    return 0;
}
