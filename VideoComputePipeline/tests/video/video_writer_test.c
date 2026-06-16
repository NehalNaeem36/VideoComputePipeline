#include "video/video_writer.h"

#include <stdio.h>

#define TEST_ASSERT(expr) do { if (!(expr)) { fprintf(stderr, "fail: %s:%d: %s\n", __FILE__, __LINE__, #expr); return 1; } } while (0)

int main(void) {
    VideoWriter writer;
    Frame frame;
    frame_init(&frame);
    TEST_ASSERT(frame_alloc(&frame, 64, 64, FRAME_FORMAT_RGB24) == 0);
    for (int y = 0; y < frame.height; ++y) {
        for (int x = 0; x < frame.width; ++x) {
            unsigned char *p = frame.data + (size_t)y * frame.stride + x * 3;
            p[0] = (unsigned char)(x * 4);
            p[1] = (unsigned char)(y * 4);
            p[2] = 64;
        }
    }

    TEST_ASSERT(video_writer_open_with_options(&writer, "data/output/video_writer_test.mp4", 64, 64, 30.0, 1, "libx264", 0) == 0);
    TEST_ASSERT(video_writer_write_frame(&writer, &frame) == 0);
    TEST_ASSERT(video_writer_flush(&writer) == 0);
    video_writer_close(&writer);
    frame_free(&frame);
    return 0;
}
