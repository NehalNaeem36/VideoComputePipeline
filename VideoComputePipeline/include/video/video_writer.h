#ifndef VIDEOCOMPUTEPIPELINE_VIDEO_WRITER_H
#define VIDEOCOMPUTEPIPELINE_VIDEO_WRITER_H

#include "core/frame.h"

typedef struct {
    void *format_ctx;
    void *codec_ctx;
    void *stream;
    void *packet;
    void *rgb_frame;
    void *yuv_frame;
    void *sws_ctx;
    int width;
    int height;
    double fps;
    int next_pts;
    int is_open;
} VideoWriter;

int video_writer_open(VideoWriter *writer, const char *output_path, int width, int height, double fps);
int video_writer_write_frame(VideoWriter *writer, const Frame *frame);
int video_writer_flush(VideoWriter *writer);
void video_writer_close(VideoWriter *writer);

#endif
