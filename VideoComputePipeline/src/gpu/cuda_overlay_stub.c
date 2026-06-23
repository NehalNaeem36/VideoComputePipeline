/*
 * CUDA overlay stub module: provides the same overlay API when hardware video
 * support is not built. Pipeline code can link consistently and report a clear
 * runtime error instead of depending on CUDA symbols.
 */
#include "gpu/cuda_overlay.h"

static const char *g_last_error = "CUDA overlay backend was not built";

int cuda_overlay_draw_nv12_boxes(CudaNV12Frame *frame,
                                 const DetectionResult *detections,
                                 int thickness,
                                 float min_confidence,
                                 int class_filter,
                                 void *cuda_stream,
                                 double *overlay_ms) {
    (void)frame;
    (void)detections;
    (void)thickness;
    (void)min_confidence;
    (void)class_filter;
    (void)cuda_stream;
    if (overlay_ms) {
        *overlay_ms = 0.0;
    }
    g_last_error = "CUDA overlay backend was not built";
    return -1;
}

const char *cuda_overlay_last_error(void) {
    return g_last_error;
}
