__kernel void grayscale_rgb24(__global const uchar *input,
                              __global uchar *output,
                              const uint width,
                              const uint height,
                              const uint stride) {
    const uint x = get_global_id(0);
    const uint y = get_global_id(1);
    if (x >= width || y >= height) {
        return;
    }

    const uint offset = y * stride + x * 3;
    const uchar r = input[offset + 0];
    const uchar g = input[offset + 1];
    const uchar b = input[offset + 2];
    const uchar gray = (uchar)(0.299f * (float)r + 0.587f * (float)g + 0.114f * (float)b + 0.5f);
    output[offset + 0] = gray;
    output[offset + 1] = gray;
    output[offset + 2] = gray;
}
