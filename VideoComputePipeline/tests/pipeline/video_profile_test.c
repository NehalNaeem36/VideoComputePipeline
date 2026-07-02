#include "pipeline/video_profile.h"

#include <stdio.h>

#define TEST_ASSERT(expr) \
    do { \
        if (!(expr)) { \
            fprintf(stderr, "Assertion failed: %s:%d: %s\n", __FILE__, __LINE__, #expr); \
            return 1; \
        } \
    } while (0)

int main(void) {
    VideoInfo info;
    VideoProfile profile;

    info.width = 3840;
    info.height = 2160;
    info.fps = 60.0;
    info.total_frames = 600;
    info.duration_sec = 10.0;

    TEST_ASSERT(video_profile_from_info /* module: pipeline/video_profile */ (&profile, &info, FRAME_FORMAT_NV12, 0, 0) == 0);
    TEST_ASSERT(profile.frame_bytes == (size_t)3840u * 2160u * 3u / 2u);
    TEST_ASSERT(profile.requires_raw_upload == 0);
    TEST_ASSERT(profile.requires_raw_download == 0);

    TEST_ASSERT(video_profile_from_info /* module: pipeline/video_profile */ (&profile, &info, FRAME_FORMAT_RGB24, 1, 1) == 0);
    TEST_ASSERT(profile.frame_bytes == (size_t)3840u * 2160u * 3u);
    TEST_ASSERT(profile.requires_raw_upload == 1);
    TEST_ASSERT(profile.requires_raw_download == 1);

    info.width = 1920;
    info.height = 1080;
    TEST_ASSERT(video_profile_from_info /* module: pipeline/video_profile */ (&profile, &info, FRAME_FORMAT_GRAY8, 1, 0) == 0);
    TEST_ASSERT(profile.frame_bytes == (size_t)1920u * 1080u);
    return 0;
}
