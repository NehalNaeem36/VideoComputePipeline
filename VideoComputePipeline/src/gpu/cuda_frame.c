/*
 * CUDA frame module: describes GPU-resident NV12 frame metadata without owning
 * FFmpeg or CUDA APIs in the public C interface. Hardware video, CUDA overlay,
 * and TensorRT inference share this structure to avoid raw CPU frame copies.
 */
#include "gpu/cuda_frame.h"

#include <string.h>

void cuda_nv12_frame_init(CudaNV12Frame *frame) {
    if (!frame) {
        return;
    }
    memset(frame, 0, sizeof(*frame));
}

void cuda_nv12_frame_clear(CudaNV12Frame *frame) {
    if (!frame) {
        return;
    }
    cuda_nv12_frame_init /* module: gpu/cuda_frame */ (frame);
}

int cuda_nv12_frame_is_valid(const CudaNV12Frame *frame) {
    return frame &&
           frame->width > 0 &&
           frame->height > 0 &&
           frame->d_y != NULL &&
           frame->d_uv != NULL &&
           frame->y_pitch > 0 &&
           frame->uv_pitch > 0;
}
