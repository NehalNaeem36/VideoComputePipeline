/*
 * Frame slot module: wraps an in-progress Frame with pipeline metadata for
 * staged processing. It is used where a reusable container is simpler than
 * passing raw Frame pointers through pipeline code.
 */
#include "pipeline/frame_slot.h"

void frame_slot_init(FrameSlot *slot) {
    if (!slot) {
        return;
    }

    frame_init /* module: core/frame */ (&slot->frame);
    slot->occupied = 0;
    slot->processing = 0;
}

void frame_slot_free(FrameSlot *slot) {
    if (!slot) {
        return;
    }

    frame_free /* module: core/frame */ (&slot->frame);
    slot->occupied = 0;
    slot->processing = 0;
}

int frame_slot_put(FrameSlot *slot, Frame *frame) {
    if (!slot || !frame || slot->occupied) {
        return -1;
    }

    if (frame_move /* module: core/frame */ (&slot->frame, frame) != 0) {
        return -1;
    }

    slot->occupied = 1;
    slot->processing = 0;
    return 0;
}

int frame_slot_take(FrameSlot *slot, Frame *out_frame) {
    if (!slot || !out_frame || !slot->occupied) {
        return -1;
    }

    if (frame_move /* module: core/frame */ (out_frame, &slot->frame) != 0) {
        return -1;
    }

    slot->occupied = 0;
    slot->processing = 0;
    return 0;
}
