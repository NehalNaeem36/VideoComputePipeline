#ifndef VIDEOCOMPUTEPIPELINE_PIPELINE_FRAME_QUEUE_H
#define VIDEOCOMPUTEPIPELINE_PIPELINE_FRAME_QUEUE_H

#include "core/frame.h"
#include <stddef.h>

/**
 * Thread-safe frame queue
 */
typedef struct {
    Frame **frames;
    size_t capacity;
    size_t head;
    size_t tail;
    size_t count;
} FrameQueue;

/**
 * Create frame queue with given capacity
 */
FrameQueue* frame_queue_create(size_t capacity);

/**
 * Destroy frame queue
 */
void frame_queue_destroy(FrameQueue *queue);

/**
 * Enqueue frame
 */
int frame_queue_enqueue(FrameQueue *queue, Frame *frame);

/**
 * Dequeue frame
 */
Frame* frame_queue_dequeue(FrameQueue *queue);

/**
 * Check if queue is empty
 */
int frame_queue_is_empty(FrameQueue *queue);

/**
 * Get queue size
 */
size_t frame_queue_size(FrameQueue *queue);

#endif // VIDEOCOMPUTEPIPELINE_PIPELINE_FRAME_QUEUE_H
