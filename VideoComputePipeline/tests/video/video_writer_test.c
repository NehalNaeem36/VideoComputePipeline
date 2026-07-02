#include "video/video_writer.h"

#include <stdio.h>

#define TEST_ASSERT(expr) do { if (!(expr)) { fprintf(stderr, "fail: %s:%d: %s\n", __FILE__, __LINE__, #expr); return 1; } } while (0)

int main(void) {
    VideoWriter writer;
    Frame frame;
    frame_init /* module: core/frame */ (&frame);
    TEST_ASSERT(frame_alloc /* module: core/frame */ (&frame, 64, 64, FRAME_FORMAT_RGB24) == 0);
    for (int y = 0; y < frame.height; ++y) {
        for (int x = 0; x < frame.width; ++x) {
            unsigned char *p = frame.data + (size_t)y * frame.stride + x * 3;
            p[0] = (unsigned char)(x * 4);
            p[1] = (unsigned char)(y * 4);
            p[2] = 64;
        }
    }

    TEST_ASSERT(video_writer_open_with_options /* module: video/video_writer */ (&writer, "data/output/video_writer_test.mp4", 64, 64, 30.0, 1, "mpeg4", 0) == 0);
    TEST_ASSERT(video_writer_write_frame /* module: video/video_writer */ (&writer, &frame) == 0);
    TEST_ASSERT(video_writer_flush /* module: video/video_writer */ (&writer) == 0);
    video_writer_close /* module: video/video_writer */ (&writer);
    frame_free /* module: core/frame */ (&frame);
    return 0;
}
