#ifndef VIDEOCOMPUTEPIPELINE_VIDEO_VIDEO_READER_H
#define VIDEOCOMPUTEPIPELINE_VIDEO_VIDEO_READER_H

#include "core/frame.h"

/**
 * VideoReader structure for FFmpeg-based video decoding
 */
typedef struct {
    const char *filename;
    uint32_t width;
    uint32_t height;
    double fps;
    uint64_t total_frames;
} VideoReader;

/**
 * Open video file and create VideoReader
 */
VideoReader* video_reader_open(const char *filename);

/**
 * Read next frame from video
 */
Frame* video_reader_read_frame(VideoReader *reader);

/**
 * Close video file and free resources
 */
void video_reader_close(VideoReader *reader);

/**
 * Get video properties
 */
void video_reader_get_properties(VideoReader *reader, uint32_t *width, uint32_t *height, double *fps);

#endif // VIDEOCOMPUTEPIPELINE_VIDEO_VIDEO_READER_H
