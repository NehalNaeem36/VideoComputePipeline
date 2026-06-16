#ifndef VIDEOCOMPUTEPIPELINE_VIDEO_READER_H
#define VIDEOCOMPUTEPIPELINE_VIDEO_READER_H

#include "core/frame.h"

#include <stdint.h>

typedef struct {
    int width;
    int height;
    double fps;
    int64_t total_frames;
    double duration_sec;
} VideoInfo;

typedef struct {
    void *format_ctx;
    void *codec_ctx;
    void *stream;
    void *packet;
    void *decoded_frame;
    void *sws_ctx;
    int video_stream_index;
    int next_frame_index;
    int decoder_flushed;
    VideoInfo info;
} VideoReader;

int video_reader_open(VideoReader *reader, const char *input_path);
int video_reader_read_frame(VideoReader *reader, Frame *out_frame);
void video_reader_close(VideoReader *reader);
const VideoInfo *video_reader_get_info(const VideoReader *reader);

#endif
