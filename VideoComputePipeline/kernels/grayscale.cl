// OpenCL kernel for grayscale conversion
// RGB24 to grayscale using standard luminosity formula

__kernel void grayscale(__global const uchar *input,
                        __global uchar *output,
                        const uint width,
                        const uint height,
                        const uint stride) {
    // TODO: Implement grayscale kernel
    // Get pixel position
    // Read RGB values
    // Convert to grayscale: Y = 0.299*R + 0.587*G + 0.114*B
    // Write output
}
