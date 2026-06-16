#ifndef VIDEOCOMPUTEPIPELINE_FRAME_SLOT_H
#define VIDEOCOMPUTEPIPELINE_FRAME_SLOT_H

#include "core/frame.h"

typedef struct {
    Frame frame;
    int occupied;
    int processing;
} FrameSlot;

void frame_slot_init(FrameSlot *slot);
void frame_slot_free(FrameSlot *slot);
int frame_slot_put(FrameSlot *slot, Frame *frame);
int frame_slot_take(FrameSlot *slot, Frame *out_frame);

#endif
