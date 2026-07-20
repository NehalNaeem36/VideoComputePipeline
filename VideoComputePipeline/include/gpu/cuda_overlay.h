#ifndef VIDEOCOMPUTEPIPELINE_GPU_CUDA_OVERLAY_H
#define VIDEOCOMPUTEPIPELINE_GPU_CUDA_OVERLAY_H

#include "gpu/cuda_frame.h"
#include "inference/detection_result.h"

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    void *d_detections;
    void *start_event;
    void *stop_event;
    size_t capacity;
} CudaOverlayContext;

void cuda_overlay_context_init(CudaOverlayContext *ctx);
int cuda_overlay_context_alloc(CudaOverlayContext *ctx, size_t max_detections);
void cuda_overlay_context_free(CudaOverlayContext *ctx);
int cuda_overlay_draw_nv12_boxes_with_context(CudaOverlayContext *ctx,
                                              CudaNV12Frame *frame,
                                              const DetectionResult *detections,
                                              int thickness,
                                              float min_confidence,
                                              int class_filter,
                                              void *cuda_stream,
                                              double *overlay_ms);
int cuda_overlay_draw_nv12_boxes(CudaNV12Frame *frame,
                                 const DetectionResult *detections,
                                 int thickness,
                                 float min_confidence,
                                 int class_filter,
                                 void *cuda_stream,
                                 double *overlay_ms);
const char *cuda_overlay_last_error(void);

#ifdef __cplusplus
}
#endif

#endif
