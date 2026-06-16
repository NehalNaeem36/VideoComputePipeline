static int clamp_int_kernel(int value, int min_value, int max_value) {
    return value < min_value ? min_value : (value > max_value ? max_value : value);
}

__kernel void blur3x3_rgb24(__global const uchar *input,
                            __global uchar *output,
                            const uint width,
                            const uint height,
                            const uint stride) {
    const uint x = get_global_id(0);
    const uint y = get_global_id(1);
    if (x >= width || y >= height) {
        return;
    }

    int sum_r = 0;
    int sum_g = 0;
    int sum_b = 0;

    for (int ky = -1; ky <= 1; ++ky) {
        const int sy = clamp_int_kernel((int)y + ky, 0, (int)height - 1);
        for (int kx = -1; kx <= 1; ++kx) {
            const int sx = clamp_int_kernel((int)x + kx, 0, (int)width - 1);
            const uint offset = (uint)sy * stride + (uint)sx * 3u;
            sum_r += input[offset + 0];
            sum_g += input[offset + 1];
            sum_b += input[offset + 2];
        }
    }

    const uint out = y * stride + x * 3u;
    output[out + 0] = (uchar)(sum_r / 9);
    output[out + 1] = (uchar)(sum_g / 9);
    output[out + 2] = (uchar)(sum_b / 9);
}
