#include "cpu/cpu_filters.h"

#include <string.h>

static int ensure_rgb_output(const Frame *input, Frame *output) {
    if (!frame_is_valid(input) || !output || input->format != FRAME_FORMAT_RGB24) {
        return -1;
    }

    if (!frame_is_valid(output) ||
        output->width != input->width ||
        output->height != input->height ||
        output->format != FRAME_FORMAT_RGB24) {
        if (frame_alloc(output, input->width, input->height, FRAME_FORMAT_RGB24) != 0) {
            return -1;
        }
    }

    output->index = input->index;
    return 0;
}

static int clamp_int(int value, int min_value, int max_value) {
    if (value < min_value) {
        return min_value;
    }
    if (value > max_value) {
        return max_value;
    }
    return value;
}

int cpu_grayscale(const Frame *input, Frame *output) {
    if (ensure_rgb_output(input, output) != 0) {
        return -1;
    }

    for (int y = 0; y < input->height; ++y) {
        const unsigned char *src = input->data + (size_t)y * input->stride;
        unsigned char *dst = output->data + (size_t)y * output->stride;

        for (int x = 0; x < input->width; ++x) {
            const unsigned char r = src[x * 3 + 0];
            const unsigned char g = src[x * 3 + 1];
            const unsigned char b = src[x * 3 + 2];
            const unsigned char gray = (unsigned char)(0.299 * r + 0.587 * g + 0.114 * b + 0.5);
            dst[x * 3 + 0] = gray;
            dst[x * 3 + 1] = gray;
            dst[x * 3 + 2] = gray;
        }
    }

    return 0;
}

static int cpu_box_blur(const Frame *input, Frame *output, int kernel_size) {
    if (ensure_rgb_output(input, output) != 0) {
        return -1;
    }

    if (kernel_size <= 0) {
        return -1;
    }

    const int start = -(kernel_size / 2);
    const int end = start + kernel_size;
    const int area = kernel_size * kernel_size;

    for (int y = 0; y < input->height; ++y) {
        unsigned char *dst_row = output->data + (size_t)y * output->stride;

        for (int x = 0; x < input->width; ++x) {
            int sum_r = 0;
            int sum_g = 0;
            int sum_b = 0;

            for (int ky = start; ky < end; ++ky) {
                const int sy = clamp_int(y + ky, 0, input->height - 1);
                const unsigned char *src_row = input->data + (size_t)sy * input->stride;

                for (int kx = start; kx < end; ++kx) {
                    const int sx = clamp_int(x + kx, 0, input->width - 1);
                    const unsigned char *pixel = src_row + sx * 3;
                    sum_r += pixel[0];
                    sum_g += pixel[1];
                    sum_b += pixel[2];
                }
            }

            dst_row[x * 3 + 0] = (unsigned char)(sum_r / area);
            dst_row[x * 3 + 1] = (unsigned char)(sum_g / area);
            dst_row[x * 3 + 2] = (unsigned char)(sum_b / area);
        }
    }

    return 0;
}

int cpu_blur3x3(const Frame *input, Frame *output) {
    return cpu_box_blur(input, output, 3);
}

int cpu_blur5x5(const Frame *input, Frame *output) {
    return cpu_box_blur(input, output, 5);
}

int cpu_blur9x9(const Frame *input, Frame *output) {
    return cpu_box_blur(input, output, 9);
}

int cpu_blur13x13(const Frame *input, Frame *output) {
    return cpu_box_blur(input, output, 13);
}
