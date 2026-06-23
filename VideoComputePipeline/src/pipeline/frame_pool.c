/*
 * Frame pool module: preallocates reusable CPU Frame objects to keep heap growth
 * bounded during long video runs. Pipeline stages acquire/release frames through
 * this pool instead of allocating per frame.
 */
#include "pipeline/frame_pool.h"

#include <stdlib.h>
#include <string.h>

int frame_pool_init(FramePool *pool, size_t capacity, int width, int height, FrameFormat format) {
    if (!pool || capacity == 0 || width <= 0 || height <= 0) {
        return -1;
    }

    memset(pool, 0, sizeof(*pool));
#ifdef _WIN32
    InitializeCriticalSection(&pool->lock);
    InitializeConditionVariable(&pool->not_empty);
#else
    pthread_mutex_init(&pool->lock, NULL);
    pthread_cond_init(&pool->not_empty, NULL);
#endif
    pool->lock_initialized = 1;

    pool->slots = (FrameSlot *)calloc(capacity, sizeof(*pool->slots));
    if (!pool->slots) {
        frame_pool_free /* module: pipeline/frame_pool */ (pool);
        return -1;
    }

    pool->capacity = capacity;
    for (size_t i = 0; i < capacity; ++i) {
        frame_slot_init /* module: pipeline/frame_slot */ (&pool->slots[i]);
        if (frame_alloc /* module: core/frame */ (&pool->slots[i].frame, width, height, format) != 0) {
            frame_pool_free /* module: pipeline/frame_pool */ (pool);
            return -1;
        }
        pool->slots[i].occupied = 1;
        pool->available++;
    }
    return 0;
}

int frame_pool_acquire(FramePool *pool, Frame *out_frame) {
    if (!pool || !out_frame) {
        return -1;
    }

#ifdef _WIN32
    if (!pool->lock_initialized) {
        return -1;
    }
    EnterCriticalSection(&pool->lock);
    while (!pool->closed && pool->available == 0) {
        SleepConditionVariableCS(&pool->not_empty, &pool->lock, INFINITE);
    }
    if (pool->closed) {
        LeaveCriticalSection(&pool->lock);
        return 0;
    }
    for (size_t i = 0; i < pool->capacity; ++i) {
        if (pool->slots[i].occupied) {
            const int result = frame_slot_take /* module: pipeline/frame_slot */ (&pool->slots[i], out_frame);
            if (result == 0) {
                pool->available--;
            }
            LeaveCriticalSection(&pool->lock);
            return result == 0 ? 1 : -1;
        }
    }
    LeaveCriticalSection(&pool->lock);
#else
    if (!pool->lock_initialized) {
        return -1;
    }
    pthread_mutex_lock(&pool->lock);
    while (!pool->closed && pool->available == 0) {
        pthread_cond_wait(&pool->not_empty, &pool->lock);
    }
    if (pool->closed) {
        pthread_mutex_unlock(&pool->lock);
        return 0;
    }
    for (size_t i = 0; i < pool->capacity; ++i) {
        if (pool->slots[i].occupied) {
            const int result = frame_slot_take /* module: pipeline/frame_slot */ (&pool->slots[i], out_frame);
            if (result == 0) {
                pool->available--;
            }
            pthread_mutex_unlock(&pool->lock);
            return result == 0 ? 1 : -1;
        }
    }
    pthread_mutex_unlock(&pool->lock);
#endif
    return -1;
}

int frame_pool_release(FramePool *pool, Frame *frame) {
    if (!pool || !frame || !frame_is_valid /* module: core/frame */ (frame)) {
        return -1;
    }

#ifdef _WIN32
    if (!pool->lock_initialized) {
        frame_free /* module: core/frame */ (frame);
        return -1;
    }
    EnterCriticalSection(&pool->lock);
    if (pool->closed) {
        LeaveCriticalSection(&pool->lock);
        frame_free /* module: core/frame */ (frame);
        return 0;
    }
    for (size_t i = 0; i < pool->capacity; ++i) {
        if (!pool->slots[i].occupied) {
            const int result = frame_slot_put /* module: pipeline/frame_slot */ (&pool->slots[i], frame);
            if (result == 0) {
                pool->available++;
                WakeConditionVariable(&pool->not_empty);
            }
            LeaveCriticalSection(&pool->lock);
            return result;
        }
    }
    LeaveCriticalSection(&pool->lock);
#else
    if (!pool->lock_initialized) {
        frame_free /* module: core/frame */ (frame);
        return -1;
    }
    pthread_mutex_lock(&pool->lock);
    if (pool->closed) {
        pthread_mutex_unlock(&pool->lock);
        frame_free /* module: core/frame */ (frame);
        return 0;
    }
    for (size_t i = 0; i < pool->capacity; ++i) {
        if (!pool->slots[i].occupied) {
            const int result = frame_slot_put /* module: pipeline/frame_slot */ (&pool->slots[i], frame);
            if (result == 0) {
                pool->available++;
                pthread_cond_signal(&pool->not_empty);
            }
            pthread_mutex_unlock(&pool->lock);
            return result;
        }
    }
    pthread_mutex_unlock(&pool->lock);
#endif

    frame_free /* module: core/frame */ (frame);
    return -1;
}

void frame_pool_close(FramePool *pool) {
    if (!pool) {
        return;
    }

#ifdef _WIN32
    if (!pool->lock_initialized) {
        return;
    }
    EnterCriticalSection(&pool->lock);
    pool->closed = 1;
    WakeAllConditionVariable(&pool->not_empty);
    LeaveCriticalSection(&pool->lock);
#else
    if (!pool->lock_initialized) {
        return;
    }
    pthread_mutex_lock(&pool->lock);
    pool->closed = 1;
    pthread_cond_broadcast(&pool->not_empty);
    pthread_mutex_unlock(&pool->lock);
#endif
}

void frame_pool_free(FramePool *pool) {
    if (!pool) {
        return;
    }

    frame_pool_close /* module: pipeline/frame_pool */ (pool);
    if (pool->slots) {
        for (size_t i = 0; i < pool->capacity; ++i) {
            frame_slot_free /* module: pipeline/frame_slot */ (&pool->slots[i]);
        }
    }
    free(pool->slots);
    pool->slots = NULL;
    pool->capacity = 0;
    pool->available = 0;

#ifdef _WIN32
    if (pool->lock_initialized) {
        DeleteCriticalSection(&pool->lock);
    }
#else
    if (pool->lock_initialized) {
        pthread_cond_destroy(&pool->not_empty);
        pthread_mutex_destroy(&pool->lock);
    }
#endif
    pool->lock_initialized = 0;
}

size_t frame_pool_available(FramePool *pool) {
    if (!pool) {
        return 0;
    }

    size_t available = 0;
#ifdef _WIN32
    if (!pool->lock_initialized) {
        return 0;
    }
    EnterCriticalSection(&pool->lock);
    available = pool->available;
    LeaveCriticalSection(&pool->lock);
#else
    if (!pool->lock_initialized) {
        return 0;
    }
    pthread_mutex_lock(&pool->lock);
    available = pool->available;
    pthread_mutex_unlock(&pool->lock);
#endif
    return available;
}
