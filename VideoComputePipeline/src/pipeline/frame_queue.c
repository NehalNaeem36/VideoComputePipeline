/*
 * Frame queue module: provides a bounded thread-safe queue for frame handoff
 * between decoder, processor, and writer stages. Threaded pipelines use it to
 * apply backpressure and preserve bounded memory.
 */
#include "pipeline/frame_queue.h"

#include <stdlib.h>

int frame_queue_init(FrameQueue *queue, size_t capacity) {
    if (!queue || capacity == 0) {
        return -1;
    }

    queue->items = (Frame *)calloc(capacity, sizeof(*queue->items));
    if (!queue->items) {
        return -1;
    }

    for (size_t i = 0; i < capacity; ++i) {
        frame_init /* module: core/frame */ (&queue->items[i]);
    }

    queue->capacity = capacity;
    queue->head = 0;
    queue->tail = 0;
    queue->count = 0;
    return 0;
}

void frame_queue_free(FrameQueue *queue) {
    if (!queue) {
        return;
    }

    if (queue->items) {
        for (size_t i = 0; i < queue->capacity; ++i) {
            frame_free /* module: core/frame */ (&queue->items[i]);
        }
    }

    free(queue->items);
    queue->items = NULL;
    queue->capacity = 0;
    queue->head = 0;
    queue->tail = 0;
    queue->count = 0;
}

int frame_queue_push(FrameQueue *queue, Frame *frame) {
    if (!queue || !frame || frame_queue_is_full /* module: pipeline/frame_queue */ (queue)) {
        return -1;
    }

    if (frame_move /* module: core/frame */ (&queue->items[queue->tail], frame) != 0) {
        return -1;
    }

    queue->tail = (queue->tail + 1) % queue->capacity;
    queue->count++;
    return 0;
}

int frame_queue_pop(FrameQueue *queue, Frame *out_frame) {
    if (!queue || !out_frame || frame_queue_is_empty /* module: pipeline/frame_queue */ (queue)) {
        return -1;
    }

    if (frame_move /* module: core/frame */ (out_frame, &queue->items[queue->head]) != 0) {
        return -1;
    }

    queue->head = (queue->head + 1) % queue->capacity;
    queue->count--;
    return 0;
}

int frame_queue_is_empty(const FrameQueue *queue) {
    return !queue || queue->count == 0;
}

int frame_queue_is_full(const FrameQueue *queue) {
    return queue && queue->count == queue->capacity;
}

size_t frame_queue_size(const FrameQueue *queue) {
    return queue ? queue->count : 0;
}
