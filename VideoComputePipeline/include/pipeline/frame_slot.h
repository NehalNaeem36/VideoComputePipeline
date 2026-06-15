#ifndef VIDEOCOMPUTEPIPELINE_PIPELINE_FRAME_SLOT_H
#define VIDEOCOMPUTEPIPELINE_PIPELINE_FRAME_SLOT_H

#include "core/frame.h"

/**
 * Frame slot for pipeline processing
 */
typedef struct {
    Frame *frame;
    int is_occupied;
    int is_processing;
} FrameSlot;

/**
 * Create frame slot
 */
FrameSlot* frame_slot_create(void);

/**
 * Destroy frame slot
 */
void frame_slot_destroy(FrameSlot *slot);

/**
 * Put frame into slot
 */
void frame_slot_put(FrameSlot *slot, Frame *frame);

/**
 * Get frame from slot
 */
Frame* frame_slot_get(FrameSlot *slot);

#endif // VIDEOCOMPUTEPIPELINE_PIPELINE_FRAME_SLOT_H
