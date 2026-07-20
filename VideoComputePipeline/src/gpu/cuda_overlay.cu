/*
 * CUDA overlay module: draws detection boxes directly into GPU-resident NV12
 * frames. The hardware detection runner calls this after inference/postprocess
 * and before NVENC writes annotated output.
 */
#include "gpu/cuda_overlay.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <string>

namespace {

thread_local std::string g_last_error = "no error";

__device__ int clamp_int_device(int value, int lo, int hi) {
    return value < lo ? lo : (value > hi ? hi : value);
}

__global__ void draw_boxes_y_kernel(uint8_t *y_plane,
                                    size_t y_pitch,
                                    int width,
                                    int height,
                                    const Detection *detections,
                                    int detection_count,
                                    int thickness,
                                    float min_confidence,
                                    int class_filter) {
    const int det_index = blockIdx.x;
    if (det_index >= detection_count) {
        return;
    }

    const Detection d = detections[det_index];
    if (d.confidence < min_confidence || (class_filter >= 0 && d.class_id != class_filter)) {
        return;
    }

    const int x1 = clamp_int_device((int)d.x1, 0, width - 1);
    const int y1 = clamp_int_device((int)d.y1, 0, height - 1);
    const int x2 = clamp_int_device((int)d.x2, 0, width - 1);
    const int y2 = clamp_int_device((int)d.y2, 0, height - 1);
    if (x2 <= x1 || y2 <= y1) {
        return;
    }

    const int perimeter = ((x2 - x1 + 1) * 2 + (y2 - y1 + 1) * 2) * thickness;
    for (int p = threadIdx.x; p < perimeter; p += blockDim.x) {
        int segment = p / thickness;
        const int t = p % thickness;
        int x = x1;
        int y = y1;

        const int horizontal = x2 - x1 + 1;
        const int vertical = y2 - y1 + 1;
        if (segment < horizontal) {
            x = x1 + segment;
            y = y1 + t;
        } else if (segment < horizontal * 2) {
            x = x1 + (segment - horizontal);
            y = y2 - t;
        } else if (segment < horizontal * 2 + vertical) {
            x = x1 + t;
            y = y1 + (segment - horizontal * 2);
        } else {
            x = x2 - t;
            y = y1 + (segment - horizontal * 2 - vertical);
        }

        x = clamp_int_device(x, 0, width - 1);
        y = clamp_int_device(y, 0, height - 1);
        y_plane[(size_t)y * y_pitch + (size_t)x] = 235u;
    }
}

float event_elapsed_ms(cudaEvent_t start, cudaEvent_t stop) {
    float ms = 0.0f;
    if (cudaEventElapsedTime(&ms, start, stop) != cudaSuccess) {
        return 0.0f;
    }
    return ms;
}

}  // namespace

extern "C" int cuda_overlay_draw_nv12_boxes(CudaNV12Frame *frame,
                                             const DetectionResult *detections,
                                             int thickness,
                                             float min_confidence,
                                             int class_filter,
                                             void *cuda_stream,
                                             double *overlay_ms) {
    CudaOverlayContext ctx;
    cuda_overlay_context_init(&ctx);
    if (!detections || cuda_overlay_context_alloc(&ctx, detections->capacity) != 0) {
        return -1;
    }
    const int result = cuda_overlay_draw_nv12_boxes_with_context(&ctx,
                                                                 frame,
                                                                 detections,
                                                                 thickness,
                                                                 min_confidence,
                                                                 class_filter,
                                                                 cuda_stream,
                                                                 overlay_ms);
    cuda_overlay_context_free(&ctx);
    return result;
}

extern "C" void cuda_overlay_context_init(CudaOverlayContext *ctx) {
    if (!ctx) {
        return;
    }
    ctx->d_detections = nullptr;
    ctx->start_event = nullptr;
    ctx->stop_event = nullptr;
    ctx->capacity = 0u;
}

extern "C" int cuda_overlay_context_alloc(CudaOverlayContext *ctx, size_t max_detections) {
    if (!ctx || max_detections == 0u) {
        g_last_error = "invalid CUDA overlay context allocation";
        return -1;
    }
    cuda_overlay_context_free(ctx);
    if (cudaMalloc(&ctx->d_detections, max_detections * sizeof(Detection)) != cudaSuccess ||
        cudaEventCreate(reinterpret_cast<cudaEvent_t *>(&ctx->start_event)) != cudaSuccess ||
        cudaEventCreate(reinterpret_cast<cudaEvent_t *>(&ctx->stop_event)) != cudaSuccess) {
        cuda_overlay_context_free(ctx);
        g_last_error = "failed to allocate CUDA overlay context";
        return -1;
    }
    ctx->capacity = max_detections;
    return 0;
}

extern "C" void cuda_overlay_context_free(CudaOverlayContext *ctx) {
    if (!ctx) {
        return;
    }
    if (ctx->d_detections) {
        cudaFree(ctx->d_detections);
    }
    if (ctx->start_event) {
        cudaEventDestroy(reinterpret_cast<cudaEvent_t>(ctx->start_event));
    }
    if (ctx->stop_event) {
        cudaEventDestroy(reinterpret_cast<cudaEvent_t>(ctx->stop_event));
    }
    cuda_overlay_context_init(ctx);
}

extern "C" int cuda_overlay_draw_nv12_boxes_with_context(CudaOverlayContext *ctx,
                                                          CudaNV12Frame *frame,
                                                          const DetectionResult *detections,
                                                          int thickness,
                                                          float min_confidence,
                                                          int class_filter,
                                                          void *cuda_stream,
                                                          double *overlay_ms) {
    if (overlay_ms) {
        *overlay_ms = 0.0;
    }
    if (!ctx || !ctx->d_detections || !ctx->start_event || !ctx->stop_event ||
        !cuda_nv12_frame_is_valid(frame) || !detections || thickness <= 0) {
        g_last_error = "invalid CUDA overlay arguments";
        return -1;
    }
    if (detections->count == 0) {
        return 0;
    }
    if (detections->count > ctx->capacity) {
        g_last_error = "CUDA overlay context capacity is too small";
        return -1;
    }

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream ? cuda_stream : frame->cuda_stream);
    Detection *d_detections = reinterpret_cast<Detection *>(ctx->d_detections);
    const size_t detections_bytes = detections->count * sizeof(Detection);
    if (cudaMemcpyAsync(d_detections, detections->items, detections_bytes, cudaMemcpyHostToDevice, stream) != cudaSuccess) {
        g_last_error = "failed to upload compact detections for overlay";
        return -1;
    }

    cudaEvent_t start = reinterpret_cast<cudaEvent_t>(ctx->start_event);
    cudaEvent_t stop = reinterpret_cast<cudaEvent_t>(ctx->stop_event);
    cudaEventRecord(start, stream);
    draw_boxes_y_kernel<<<(unsigned int)detections->count, 128, 0, stream>>>(frame->d_y,
                                                                             frame->y_pitch,
                                                                             frame->width,
                                                                             frame->height,
                                                                             d_detections,
                                                                             (int)detections->count,
                                                                             thickness,
                                                                             min_confidence,
                                                                             class_filter);
    const cudaError_t launch_result = cudaGetLastError();
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);
    if (overlay_ms) {
        *overlay_ms = (double)event_elapsed_ms(start, stop);
    }
    if (launch_result != cudaSuccess) {
        g_last_error = cudaGetErrorString(launch_result);
        return -1;
    }
    return 0;
}

extern "C" const char *cuda_overlay_last_error(void) {
    return g_last_error.c_str();
}
