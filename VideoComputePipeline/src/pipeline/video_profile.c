#include "pipeline/video_profile.h"

#include <string.h>

void video_profile_init(VideoProfile *profile) {
    if (!profile) {
        return;
    }
    memset(profile, 0, sizeof(*profile));
    profile->working_format = FRAME_FORMAT_RGB24;
}

static size_t frame_bytes_for_format(int width, int height, FrameFormat format) {
    const size_t pixels = (size_t)width * (size_t)height;
    switch (format) {
        case FRAME_FORMAT_NV12:
            return pixels * 3u / 2u;
        case FRAME_FORMAT_GRAY8:
            return pixels;
        case FRAME_FORMAT_RGB24:
        default:
            return pixels * 3u;
    }
}

int video_profile_from_info(VideoProfile *profile,
                            const VideoInfo *info,
                            FrameFormat working_format,
                            int requires_raw_upload,
                            int requires_raw_download) {
    if (!profile || !info || info->width <= 0 || info->height <= 0) {
        return -1;
    }

    video_profile_init(profile);
    profile->width = info->width;
    profile->height = info->height;
    profile->fps = info->fps;
    profile->total_frames = info->total_frames;
    profile->duration_sec = info->duration_sec;
    profile->working_format = working_format;
    profile->frame_bytes = frame_bytes_for_format(info->width, info->height, working_format);
    profile->requires_raw_upload = requires_raw_upload ? 1 : 0;
    profile->requires_raw_download = requires_raw_download ? 1 : 0;
    return profile->frame_bytes > 0u ? 0 : -1;
}
