#include "pipeline/frame_batch.h"

#include <stdlib.h>
#include <string.h>

void frame_batch_init(FrameBatch *batch) {
    if (!batch) {
        return;
    }
    memset(batch, 0, sizeof(*batch));
}

int frame_batch_alloc(FrameBatch *batch, int capacity, int use_cuda_frames, size_t max_detections_per_frame) {
    if (!batch || capacity <= 0 || max_detections_per_frame == 0u) {
        return -1;
    }

    frame_batch_init(batch);
    batch->capacity = capacity;
    batch->use_cuda_frames = use_cuda_frames ? 1 : 0;
    batch->detections = (DetectionResult *)calloc((size_t)capacity, sizeof(*batch->detections));
    batch->timings = (FrameTiming *)calloc((size_t)capacity, sizeof(*batch->timings));
    if (!batch->detections || !batch->timings) {
        frame_batch_free(batch);
        return -1;
    }

    if (batch->use_cuda_frames) {
        batch->cuda_frames = (CudaNV12Frame *)calloc((size_t)capacity, sizeof(*batch->cuda_frames));
        if (!batch->cuda_frames) {
            frame_batch_free(batch);
            return -1;
        }
        for (int i = 0; i < capacity; ++i) {
            cuda_nv12_frame_init(&batch->cuda_frames[i]);
        }
    } else {
        batch->cpu_frames = (Frame *)calloc((size_t)capacity, sizeof(*batch->cpu_frames));
        if (!batch->cpu_frames) {
            frame_batch_free(batch);
            return -1;
        }
        for (int i = 0; i < capacity; ++i) {
            frame_init(&batch->cpu_frames[i]);
        }
    }

    for (int i = 0; i < capacity; ++i) {
        detection_result_init(&batch->detections[i]);
        if (detection_result_alloc(&batch->detections[i], max_detections_per_frame) != 0) {
            frame_batch_free(batch);
            return -1;
        }
    }
    return 0;
}

int frame_batch_alloc_cpu_nv12_frames(FrameBatch *batch, int width, int height) {
    if (!batch || batch->use_cuda_frames || !batch->cpu_frames || width <= 0 || height <= 0) {
        return -1;
    }
    for (int i = 0; i < batch->capacity; ++i) {
        if (frame_alloc(&batch->cpu_frames[i], width, height, FRAME_FORMAT_NV12) != 0) {
            return -1;
        }
    }
    return 0;
}

void frame_batch_clear(FrameBatch *batch) {
    if (!batch) {
        return;
    }
    batch->valid_frames = 0;
    for (int i = 0; i < batch->capacity; ++i) {
        memset(&batch->timings[i], 0, sizeof(batch->timings[i]));
        if (batch->detections) {
            detection_result_clear(&batch->detections[i]);
        }
        if (batch->use_cuda_frames && batch->cuda_frames) {
            cuda_nv12_frame_clear(&batch->cuda_frames[i]);
        }
    }
}

void frame_batch_free(FrameBatch *batch) {
    if (!batch) {
        return;
    }
    if (batch->detections) {
        for (int i = 0; i < batch->capacity; ++i) {
            detection_result_free(&batch->detections[i]);
        }
    }
    if (batch->cpu_frames) {
        for (int i = 0; i < batch->capacity; ++i) {
            frame_free(&batch->cpu_frames[i]);
        }
    }
    free(batch->cpu_frames);
    free(batch->cuda_frames);
    free(batch->detections);
    free(batch->timings);
    frame_batch_init(batch);
}
