#include "video/video_reader.h"

#include <stdio.h>

#define TEST_ASSERT(expr) do { if (!(expr)) { fprintf(stderr, "fail: %s:%d: %s\n", __FILE__, __LINE__, #expr); return 1; } } while (0)

#ifndef VCP_SOURCE_DIR
#define VCP_SOURCE_DIR "."
#endif

int main(void) {
    VideoReader reader;
    const char *input_path = VCP_SOURCE_DIR "/data/input/15592600_3840_2160_60fps.mp4";
    if (video_reader_open_with_threads(&reader, input_path, 1) != 0) {
        printf("video_reader_test skipped: input video not available\n");
        return 0;
    }

    const VideoInfo *info = video_reader_get_info(&reader);
    TEST_ASSERT(info != NULL);
    TEST_ASSERT(info->width > 0);
    TEST_ASSERT(info->height > 0);

    Frame frame;
    frame_init(&frame);
    const int read_result = video_reader_read_frame(&reader, &frame);
    TEST_ASSERT(read_result == 1);
    TEST_ASSERT(frame_is_valid(&frame));
    frame_free(&frame);
    video_reader_close(&reader);
    return 0;
}
