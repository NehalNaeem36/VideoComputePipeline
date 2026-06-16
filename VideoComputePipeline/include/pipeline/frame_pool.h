#ifndef VIDEOCOMPUTEPIPELINE_FRAME_POOL_H
#define VIDEOCOMPUTEPIPELINE_FRAME_POOL_H

#include "core/frame.h"
#include "pipeline/frame_slot.h"

#include <stddef.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <pthread.h>
#endif

typedef struct {
    FrameSlot *slots;
    size_t capacity;
    size_t available;
    int closed;
    int lock_initialized;
#ifdef _WIN32
    CRITICAL_SECTION lock;
    CONDITION_VARIABLE not_empty;
#else
    pthread_mutex_t lock;
    pthread_cond_t not_empty;
#endif
} FramePool;

int frame_pool_init(FramePool *pool, size_t capacity, int width, int height, FrameFormat format);
int frame_pool_acquire(FramePool *pool, Frame *out_frame);
int frame_pool_release(FramePool *pool, Frame *frame);
void frame_pool_close(FramePool *pool);
void frame_pool_free(FramePool *pool);
size_t frame_pool_available(FramePool *pool);

#endif
