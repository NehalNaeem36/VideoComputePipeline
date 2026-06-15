#ifndef VIDEOCOMPUTEPIPELINE_VIDEO_VIDEO_WRITER_H
#define VIDEOCOMPUTEPIPELINE_VIDEO_VIDEO_WRITER_H

#include "core/frame.h"

/**
 * VideoWriter structure for FFmpeg-based video encoding
 */
typedef struct {
    const char *filename;
    uint32_t width;
    uint32_t height;
    double fps;
    const char *codec;
} VideoWriter;

/**
 * Create video file and initialize VideoWriter
 */
VideoWriter* video_writer_create(const char *filename, uint32_t width, uint32_t height, double fps, const char *codec);

/**
 * Write frame to video file
 */
int video_writer_write_frame(VideoWriter *writer, Frame *frame);

/**
 * Finalize video file and free resources
 */
void video_writer_close(VideoWriter *writer);

#endif // VIDEOCOMPUTEPIPELINE_VIDEO_VIDEO_WRITER_H
