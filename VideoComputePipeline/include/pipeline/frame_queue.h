#ifndef VIDEOCOMPUTEPIPELINE_FRAME_QUEUE_H
#define VIDEOCOMPUTEPIPELINE_FRAME_QUEUE_H

#include "core/frame.h"

#include <stddef.h>

typedef struct {
    Frame *items;
    size_t capacity;
    size_t head;
    size_t tail;
    size_t count;
} FrameQueue;

int frame_queue_init(FrameQueue *queue, size_t capacity);
void frame_queue_free(FrameQueue *queue);
int frame_queue_push(FrameQueue *queue, Frame *frame);
int frame_queue_pop(FrameQueue *queue, Frame *out_frame);
int frame_queue_is_empty(const FrameQueue *queue);
int frame_queue_is_full(const FrameQueue *queue);
size_t frame_queue_size(const FrameQueue *queue);

#endif
