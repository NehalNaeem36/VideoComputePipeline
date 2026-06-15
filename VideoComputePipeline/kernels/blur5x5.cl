// OpenCL kernel for 5x5 box blur

__kernel void blur5x5(__global const uchar *input,
                      __global uchar *output,
                      const uint width,
                      const uint height,
                      const uint stride) {
    // TODO: Implement 5x5 blur kernel
    // For each output pixel at (x, y):
    // Average 5x5 neighborhood (25 pixels)
    // Handle border cases (clamp or edge-repeat)
    // Write blurred value to output
}
