#include "video/video_reader.h"

#include <stdio.h>

#define TEST_ASSERT(expr) do { if (!(expr)) { fprintf(stderr, "fail: %s:%d: %s\n", __FILE__, __LINE__, #expr); return 1; } } while (0)

#ifndef VCP_SOURCE_DIR
#define VCP_SOURCE_DIR "."
#endif

int main(void) {
    VideoReader reader;
    const char *input_path = VCP_SOURCE_DIR "/data/input/15592600_3840_2160_60fps.mp4";
    if (video_reader_open_with_threads /* module: video/video_reader */ (&reader, input_path, 1) != 0) {
        printf("video_reader_test skipped: input video not available\n");
        return 0;
    }

    const VideoInfo *info = video_reader_get_info(&reader);
    TEST_ASSERT(info != NULL);
    TEST_ASSERT(info->width > 0);
    TEST_ASSERT(info->height > 0);

    Frame frame;
    frame_init /* module: core/frame */ (&frame);
    const int read_result = video_reader_read_frame /* module: video/video_reader */ (&reader, &frame);
    TEST_ASSERT(read_result == 1);
    TEST_ASSERT(frame_is_valid /* module: core/frame */ (&frame));
    frame_free /* module: core/frame */ (&frame);
    video_reader_close /* module: video/video_reader */ (&reader);

    if (video_reader_open_with_threads /* module: video/video_reader */ (&reader, input_path, 1) != 0) {
        printf("video_reader_test skipped NV12 read: input video not available\n");
        return 0;
    }
    frame_init /* module: core/frame */ (&frame);
    const int nv12_result = video_reader_read_frame_as /* module: video/video_reader */ (&reader, &frame, FRAME_FORMAT_NV12);
    TEST_ASSERT(nv12_result == 1);
    TEST_ASSERT(frame_is_valid /* module: core/frame */ (&frame));
    TEST_ASSERT(frame.format == FRAME_FORMAT_NV12);
    TEST_ASSERT(frame.planes[0] == frame.data);
    TEST_ASSERT(frame.planes[1] != NULL);
    frame_free /* module: core/frame */ (&frame);
    video_reader_close /* module: video/video_reader */ (&reader);
    return 0;
}
