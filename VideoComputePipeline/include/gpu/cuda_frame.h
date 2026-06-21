#ifndef VIDEOCOMPUTEPIPELINE_GPU_CUDA_FRAME_H
#define VIDEOCOMPUTEPIPELINE_GPU_CUDA_FRAME_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int index;
    int width;
    int height;
    int64_t pts;
    int64_t dts;
    double timestamp_ms;
    uint8_t *d_y;
    uint8_t *d_uv;
    size_t y_pitch;
    size_t uv_pitch;
    void *cuda_stream;
    void *av_frame;
    void *hw_frames_ctx;
    int owns_av_frame;
    int owns_cuda_memory;
} CudaNV12Frame;

void cuda_nv12_frame_init(CudaNV12Frame *frame);
void cuda_nv12_frame_clear(CudaNV12Frame *frame);
int cuda_nv12_frame_is_valid(const CudaNV12Frame *frame);

#ifdef __cplusplus
}
#endif

#endif
