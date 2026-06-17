#include "pipeline/frame_slot.h"

#include <stdio.h>

#define TEST_ASSERT(expr) do { if (!(expr)) { fprintf(stderr, "fail: %s:%d: %s\n", __FILE__, __LINE__, #expr); return 1; } } while (0)

int main(void) {
    FrameSlot slot;
    Frame in;
    Frame out;
    frame_slot_init /* module: pipeline/frame_slot */ (&slot);
    frame_init /* module: core/frame */ (&in);
    frame_init /* module: core/frame */ (&out);
    TEST_ASSERT(frame_alloc /* module: core/frame */ (&in, 1, 1, FRAME_FORMAT_RGB24) == 0);
    TEST_ASSERT(frame_slot_put /* module: pipeline/frame_slot */ (&slot, &in) == 0);
    TEST_ASSERT(!frame_is_valid /* module: core/frame */ (&in));
    TEST_ASSERT(frame_slot_take /* module: pipeline/frame_slot */ (&slot, &out) == 0);
    TEST_ASSERT(frame_is_valid /* module: core/frame */ (&out));
    frame_free /* module: core/frame */ (&out);
    frame_slot_free /* module: pipeline/frame_slot */ (&slot);
    return 0;
}
