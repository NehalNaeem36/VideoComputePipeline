#ifndef VIDEOCOMPUTEPIPELINE_VIDEO_HW_READER_H
#define VIDEOCOMPUTEPIPELINE_VIDEO_HW_READER_H

#include "gpu/cuda_frame.h"
#include "video/video_reader.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    void *format_ctx;
    void *codec_ctx;
    void *stream;
    void *packet;
    void *decoded_frame;
    void *hw_device_ctx;
    int video_stream_index;
    int next_frame_index;
    int decoder_flushed;
    VideoInfo info;
} VideoHWReader;

int video_hw_reader_open(VideoHWReader *reader, const char *path, int decoder_threads);
int video_hw_reader_read_cuda_nv12(VideoHWReader *reader, CudaNV12Frame *out);
const VideoInfo *video_hw_reader_get_info(const VideoHWReader *reader);
void video_hw_reader_release_frame(VideoHWReader *reader, CudaNV12Frame *frame);
void video_hw_reader_close(VideoHWReader *reader);
const char *video_hw_reader_last_error(void);

#ifdef __cplusplus
}
#endif

#endif
