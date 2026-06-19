#ifndef VIDEOCOMPUTEPIPELINE_GPU_CUDA_OVERLAY_H
#define VIDEOCOMPUTEPIPELINE_GPU_CUDA_OVERLAY_H

#include "gpu/cuda_frame.h"
#include "inference/detection_result.h"

#ifdef __cplusplus
extern "C" {
#endif

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
