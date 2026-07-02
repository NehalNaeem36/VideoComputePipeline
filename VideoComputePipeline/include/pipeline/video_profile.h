#ifndef VIDEOCOMPUTEPIPELINE_PIPELINE_VIDEO_PROFILE_H
#define VIDEOCOMPUTEPIPELINE_PIPELINE_VIDEO_PROFILE_H

#include "core/frame.h"
#include "video/video_reader.h"

#include <stddef.h>
#include <stdint.h>

typedef struct {
    int width;
    int height;
    double fps;
    int64_t total_frames;
    double duration_sec;
    FrameFormat working_format;
    size_t frame_bytes;
    int requires_raw_upload;
    int requires_raw_download;
} VideoProfile;

void video_profile_init(VideoProfile *profile);
int video_profile_from_info(VideoProfile *profile,
                            const VideoInfo *info,
                            FrameFormat working_format,
                            int requires_raw_upload,
                            int requires_raw_download);

#endif
