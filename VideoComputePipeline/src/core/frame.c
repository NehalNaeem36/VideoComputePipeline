#include "core/frame.h"

#include <limits.h>
#include <stdlib.h>
#include <string.h>

static int frame_channels_for_format(FrameFormat format) {
    switch (format) {
        case FRAME_FORMAT_RGB24:
            return 3;
        case FRAME_FORMAT_GRAY8:
            return 1;
        default:
            return 0;
    }
}

void frame_init(Frame *frame) {
    if (!frame) {
        return;
    }

    frame->index = 0;
    frame->width = 0;
    frame->height = 0;
    frame->channels = 0;
    frame->format = FRAME_FORMAT_RGB24;
    frame->stride = 0;
    frame->size = 0;
    frame->data = NULL;
}

int frame_alloc(Frame *frame, int width, int height, FrameFormat format) {
    if (!frame || width <= 0 || height <= 0) {
        return -1;
    }

    const int channels = frame_channels_for_format(format);
    if (channels == 0) {
        return -1;
    }

    const size_t stride = frame_calculate_stride(width, format);
    const size_t size = frame_calculate_size(width, height, format);
    if (stride == 0 || size == 0) {
        return -1;
    }

    uint8_t *data = (uint8_t *)calloc(1, size);
    if (!data) {
        return -1;
    }

    frame_free(frame);
    frame->index = 0;
    frame->width = width;
    frame->height = height;
    frame->channels = channels;
    frame->format = format;
    frame->stride = stride;
    frame->size = size;
    frame->data = data;
    return 0;
}

void frame_free(Frame *frame) {
    if (!frame) {
        return;
    }

    free(frame->data);
    frame_init(frame);
}

int frame_copy(Frame *dst, const Frame *src) {
    if (!dst || !frame_is_valid(src)) {
        return -1;
    }

    Frame copy;
    frame_init(&copy);

    if (frame_alloc(&copy, src->width, src->height, src->format) != 0) {
        return -1;
    }

    copy.index = src->index;
    memcpy(copy.data, src->data, src->size);

    frame_free(dst);
    *dst = copy;
    return 0;
}

int frame_move(Frame *dst, Frame *src) {
    if (!dst || !src) {
        return -1;
    }

    if (dst == src) {
        return 0;
    }

    frame_free(dst);
    *dst = *src;
    frame_init(src);
    return 0;
}

int frame_is_valid(const Frame *frame) {
    return frame &&
           frame->width > 0 &&
           frame->height > 0 &&
           frame->channels == frame_channels_for_format(frame->format) &&
           frame->stride > 0 &&
           frame->size > 0 &&
           frame->data != NULL;
}

size_t frame_calculate_stride(int width, FrameFormat format) {
    const int channels = frame_channels_for_format(format);
    if (width <= 0 || channels <= 0) {
        return 0;
    }

    if ((size_t)width > SIZE_MAX / (size_t)channels) {
        return 0;
    }

    return (size_t)width * (size_t)channels;
}

size_t frame_calculate_size(int width, int height, FrameFormat format) {
    if (height <= 0) {
        return 0;
    }

    const size_t stride = frame_calculate_stride(width, format);
    if (stride == 0 || (size_t)height > SIZE_MAX / stride) {
        return 0;
    }

    return stride * (size_t)height;
}
