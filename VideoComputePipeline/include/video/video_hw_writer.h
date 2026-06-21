#ifndef VIDEOCOMPUTEPIPELINE_VIDEO_HW_WRITER_H
#define VIDEOCOMPUTEPIPELINE_VIDEO_HW_WRITER_H

#include "benchmark/benchmark.h"
#include "gpu/cuda_frame.h"
#include "video/video_reader.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    char encoder_name[64];
    void *format_ctx;
    void *codec_ctx;
    void *stream;
    void *packet;
    void *hw_device_ctx;
    void *hw_frames_ctx;
    int width;
    int height;
    double fps;
    int next_pts;
    int64_t submitted_frames;
    int64_t written_packets;
    int is_open;
} VideoHWWriter;

int video_hw_writer_open(VideoHWWriter *writer,
                         const char *output_path,
                         const VideoInfo *input_info,
                         const char *encoder_name,
                         int lossless);
int video_hw_writer_write_cuda_nv12(VideoHWWriter *writer, const CudaNV12Frame *frame, FrameTiming *timing);
int video_hw_writer_flush(VideoHWWriter *writer);
void video_hw_writer_close(VideoHWWriter *writer);
const char *video_hw_writer_last_error(void);

#ifdef __cplusplus
}
#endif

#endif
