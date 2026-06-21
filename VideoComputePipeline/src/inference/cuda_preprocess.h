#ifndef VIDEOCOMPUTEPIPELINE_INFERENCE_CUDA_PREPROCESS_H
#define VIDEOCOMPUTEPIPELINE_INFERENCE_CUDA_PREPROCESS_H

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

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
                                                  cudaStream_t stream);

cudaError_t vcp_cuda_preprocess_nv12_to_nchw_fp32(const uint8_t *d_y,
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
                                                  float *d_output,
                                                  cudaStream_t stream);

#endif
