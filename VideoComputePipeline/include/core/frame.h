#ifndef VIDEOCOMPUTEPIPELINE_CORE_FRAME_H
#define VIDEOCOMPUTEPIPELINE_CORE_FRAME_H

#include <stddef.h>
#include <stdint.h>

typedef enum {
    FRAME_FORMAT_RGB24 = 0,
    FRAME_FORMAT_GRAY8 = 1,
    FRAME_FORMAT_NV12 = 2
} FrameFormat;

typedef struct {
    int index;
    int width;
    int height;
    int channels;
    FrameFormat format;
    size_t stride;
    size_t size;
    uint8_t *data;
    uint8_t *planes[4];
    size_t linesize[4];
} Frame;

void frame_init(Frame *frame);
int frame_alloc(Frame *frame, int width, int height, FrameFormat format);
void frame_free(Frame *frame);
int frame_copy(Frame *dst, const Frame *src);
int frame_move(Frame *dst, Frame *src);
int frame_is_valid(const Frame *frame);
size_t frame_calculate_stride(int width, FrameFormat format);
size_t frame_calculate_size(int width, int height, FrameFormat format);

#endif
