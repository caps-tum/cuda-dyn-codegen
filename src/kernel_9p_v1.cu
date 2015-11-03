#include <thrust/device_ptr.h>

__global__ void ninePoint1(thrust::device_ptr<float const> const input, thrust::device_ptr<float> const output, int width) {
    auto x = blockIdx.x * blockDim.x + threadIdx.x + 2;
    auto y = blockIdx.y * blockDim.y + threadIdx.y + 2;

    auto index = y * width + x;

    output[index] = 1.0f / 9 * (
        1 * input[index - width - width] +
        1 * input[index - width] +
        1 * input[index - 1 - 1] +
        1 * input[index - 1] +
        1 * input[index] +
        1 * input[index + 1] +
        1 * input[index + 1 + 1] +
        1 * input[index + width] +
        1 * input[index + width + width]
    );
}
