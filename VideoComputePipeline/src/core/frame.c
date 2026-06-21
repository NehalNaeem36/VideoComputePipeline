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
        case FRAME_FORMAT_NV12:
            return 1;
        default:
            return 0;
    }
}

static void frame_set_planes(Frame *frame) {
    if (!frame) {
        return;
    }

    memset(frame->planes, 0, sizeof(frame->planes));
    memset(frame->linesize, 0, sizeof(frame->linesize));

    if (!frame->data) {
        return;
    }

    frame->planes[0] = frame->data;
    frame->linesize[0] = frame->stride;
    if (frame->format == FRAME_FORMAT_NV12) {
        frame->planes[1] = frame->data + frame->stride * (size_t)frame->height;
        frame->linesize[1] = frame->stride;
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
    memset(frame->planes, 0, sizeof(frame->planes));
    memset(frame->linesize, 0, sizeof(frame->linesize));
}

int frame_alloc(Frame *frame, int width, int height, FrameFormat format) {
    if (!frame || width <= 0 || height <= 0) {
        return -1;
    }

    const int channels = frame_channels_for_format /* module: core/frame */ (format);
    if (channels == 0) {
        return -1;
    }

    const size_t stride = frame_calculate_stride /* module: core/frame */ (width, format);
    const size_t size = frame_calculate_size /* module: core/frame */ (width, height, format);
    if (stride == 0 || size == 0) {
        return -1;
    }

    uint8_t *data = (uint8_t *)calloc(1, size);
    if (!data) {
        return -1;
    }

    frame_free /* module: core/frame */ (frame);
    frame->index = 0;
    frame->width = width;
    frame->height = height;
    frame->channels = channels;
    frame->format = format;
    frame->stride = stride;
    frame->size = size;
    frame->data = data;
    frame_set_planes /* module: core/frame */ (frame);
    return 0;
}

void frame_free(Frame *frame) {
    if (!frame) {
        return;
    }

    free(frame->data);
    frame_init /* module: core/frame */ (frame);
}

int frame_copy(Frame *dst, const Frame *src) {
    if (!dst || !frame_is_valid /* module: core/frame */ (src)) {
        return -1;
    }

    Frame copy;
    frame_init /* module: core/frame */ (&copy);

    if (frame_alloc /* module: core/frame */ (&copy, src->width, src->height, src->format) != 0) {
        return -1;
    }

    copy.index = src->index;
    memcpy(copy.data, src->data, src->size);

    frame_free /* module: core/frame */ (dst);
    *dst = copy;
    frame_set_planes /* module: core/frame */ (dst);
    return 0;
}

int frame_move(Frame *dst, Frame *src) {
    if (!dst || !src) {
        return -1;
    }

    if (dst == src) {
        return 0;
    }

    frame_free /* module: core/frame */ (dst);
    *dst = *src;
    frame_init /* module: core/frame */ (src);
    frame_set_planes /* module: core/frame */ (dst);
    return 0;
}

int frame_is_valid(const Frame *frame) {
    return frame &&
           frame->width > 0 &&
           frame->height > 0 &&
           frame->channels == frame_channels_for_format /* module: core/frame */ (frame->format) &&
           frame->stride > 0 &&
           frame->size > 0 &&
           frame->data != NULL &&
           frame->planes[0] == frame->data &&
           frame->linesize[0] == frame->stride &&
           (frame->format != FRAME_FORMAT_NV12 || (frame->planes[1] != NULL && frame->linesize[1] == frame->stride));
}

size_t frame_calculate_stride(int width, FrameFormat format) {
    if (format == FRAME_FORMAT_NV12) {
        return width > 0 && (width % 2) == 0 ? (size_t)width : 0;
    }

    const int channels = frame_channels_for_format /* module: core/frame */ (format);
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

    const size_t stride = frame_calculate_stride /* module: core/frame */ (width, format);
    if (stride == 0 || (size_t)height > SIZE_MAX / stride) {
        return 0;
    }

    if (format == FRAME_FORMAT_NV12) {
        if ((height % 2) != 0) {
            return 0;
        }
        const size_t y_size = stride * (size_t)height;
        if (y_size > SIZE_MAX / 3u) {
            return 0;
        }
        return y_size + y_size / 2u;
    }

    return stride * (size_t)height;
}
