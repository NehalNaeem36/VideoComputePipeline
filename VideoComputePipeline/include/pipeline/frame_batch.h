#ifndef VIDEOCOMPUTEPIPELINE_PIPELINE_FRAME_BATCH_H
#define VIDEOCOMPUTEPIPELINE_PIPELINE_FRAME_BATCH_H

#include "benchmark/benchmark.h"
#include "core/frame.h"
#include "gpu/cuda_frame.h"
#include "inference/detection_result.h"

#include <stddef.h>

typedef struct {
    int capacity;
    int valid_frames;
    int use_cuda_frames;
    Frame *cpu_frames;
    CudaNV12Frame *cuda_frames;
    DetectionResult *detections;
    FrameTiming *timings;
} FrameBatch;

void frame_batch_init(FrameBatch *batch);
int frame_batch_alloc(FrameBatch *batch, int capacity, int use_cuda_frames, size_t max_detections_per_frame);
int frame_batch_alloc_cpu_nv12_frames(FrameBatch *batch, int width, int height);
void frame_batch_clear(FrameBatch *batch);
void frame_batch_free(FrameBatch *batch);

#endif
