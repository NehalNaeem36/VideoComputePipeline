#include "cuda_preprocess.h"

#include <math.h>

static __device__ float clamp_float(float value, float lo, float hi) {
    return fminf(fmaxf(value, lo), hi);
}

static __device__ int clamp_int(int value, int lo, int hi) {
    return value < lo ? lo : (value > hi ? hi : value);
}

static __device__ float sample_plane_bilinear(const uint8_t *plane,
                                              int width,
                                              int height,
                                              size_t stride,
                                              float x,
                                              float y) {
    x = clamp_float(x, 0.0f, (float)(width - 1));
    y = clamp_float(y, 0.0f, (float)(height - 1));

    const int x0 = (int)floorf(x);
    const int y0 = (int)floorf(y);
    const int x1 = clamp_int(x0 + 1, 0, width - 1);
    const int y1 = clamp_int(y0 + 1, 0, height - 1);
    const float fx = x - (float)x0;
    const float fy = y - (float)y0;

    const float p00 = (float)plane[y0 * stride + x0];
    const float p10 = (float)plane[y0 * stride + x1];
    const float p01 = (float)plane[y1 * stride + x0];
    const float p11 = (float)plane[y1 * stride + x1];
    const float top = p00 + (p10 - p00) * fx;
    const float bottom = p01 + (p11 - p01) * fx;
    return top + (bottom - top) * fy;
}

static __global__ void nv12_to_nchw_fp16_kernel(const uint8_t *y_plane,
                                                const uint8_t *uv_plane,
                                                int src_width,
                                                int src_height,
                                                size_t y_stride,
                                                size_t uv_stride,
                                                int dst_width,
                                                int dst_height,
                                                float scale,
                                                float pad_x,
                                                float pad_y,
                                                __half *output) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dst_width || y >= dst_height) {
        return;
    }

    float r = 114.0f;
    float g = 114.0f;
    float b = 114.0f;

    const float src_x = ((float)x - pad_x) / scale;
    const float src_y = ((float)y - pad_y) / scale;
    if (src_x >= 0.0f && src_y >= 0.0f && src_x <= (float)(src_width - 1) && src_y <= (float)(src_height - 1)) {
        const float yy = sample_plane_bilinear(y_plane, src_width, src_height, y_stride, src_x, src_y);
        const int uv_x = clamp_int(((int)src_x) & ~1, 0, src_width - 2);
        const int uv_y = clamp_int((int)(src_y * 0.5f), 0, src_height / 2 - 1);
        const uint8_t *uv = uv_plane + uv_y * uv_stride + uv_x;
        const float u = (float)uv[0] - 128.0f;
        const float v = (float)uv[1] - 128.0f;

        r = yy + 1.402f * v;
        g = yy - 0.344136f * u - 0.714136f * v;
        b = yy + 1.772f * u;
        r = clamp_float(r, 0.0f, 255.0f);
        g = clamp_float(g, 0.0f, 255.0f);
        b = clamp_float(b, 0.0f, 255.0f);
    }

    const int index = y * dst_width + x;
    const int plane_size = dst_width * dst_height;
    output[index] = __float2half(r / 255.0f);
    output[plane_size + index] = __float2half(g / 255.0f);
    output[plane_size * 2 + index] = __float2half(b / 255.0f);
}

cudaError_t vcp_cuda_preprocess_nv12_to_nchw_fp16(const uint8_t *d_y,
                                                  const uint8_t *d_uv,
                                                  int src_width,
                                                  int src_height,
                                                  size_t src_y_stride,
                                                  size_t src_uv_stride,
                                                  int dst_width,
                                                  int dst_height,
                                                  float scale,
                                                  float pad_x,
                                                  float pad_y,
                                                  __half *d_output,
                                                  cudaStream_t stream) {
    if (!d_y || !d_uv || !d_output || src_width <= 0 || src_height <= 0 || dst_width <= 0 || dst_height <= 0) {
        return cudaErrorInvalidValue;
    }

    const dim3 block(16, 16);
    const dim3 grid((unsigned int)((dst_width + block.x - 1) / block.x),
                    (unsigned int)((dst_height + block.y - 1) / block.y));
    nv12_to_nchw_fp16_kernel<<<grid, block, 0, stream>>>(d_y,
                                                         d_uv,
                                                         src_width,
                                                         src_height,
                                                         src_y_stride,
                                                         src_uv_stride,
                                                         dst_width,
                                                         dst_height,
                                                         scale,
                                                         pad_x,
                                                         pad_y,
                                                         d_output);
    return cudaGetLastError();
}
