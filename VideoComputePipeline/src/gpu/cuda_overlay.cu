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
    if (overlay_ms) {
        *overlay_ms = 0.0;
    }
    if (!cuda_nv12_frame_is_valid(frame) || !detections || thickness <= 0) {
        g_last_error = "invalid CUDA overlay arguments";
        return -1;
    }
    if (detections->count == 0) {
        return 0;
    }

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream ? cuda_stream : frame->cuda_stream);
    Detection *d_detections = nullptr;
    const size_t detections_bytes = detections->count * sizeof(Detection);
    if (cudaMalloc(&d_detections, detections_bytes) != cudaSuccess ||
        cudaMemcpyAsync(d_detections, detections->items, detections_bytes, cudaMemcpyHostToDevice, stream) != cudaSuccess) {
        if (d_detections) {
            cudaFree(d_detections);
        }
        g_last_error = "failed to upload compact detections for overlay";
        return -1;
    }

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    if (cudaEventCreate(&start) != cudaSuccess || cudaEventCreate(&stop) != cudaSuccess) {
        if (start) cudaEventDestroy(start);
        if (stop) cudaEventDestroy(stop);
        cudaFree(d_detections);
        g_last_error = "failed to create CUDA overlay events";
        return -1;
    }

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
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_detections);
    if (launch_result != cudaSuccess) {
        g_last_error = cudaGetErrorString(launch_result);
        return -1;
    }
    return 0;
}

extern "C" const char *cuda_overlay_last_error(void) {
    return g_last_error.c_str();
}
