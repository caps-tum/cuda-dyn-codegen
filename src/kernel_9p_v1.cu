#include <thrust/device_ptr.h>

__global__ void ninePoint1(thrust::device_ptr<float const> const input, thrust::device_ptr<float> const output, int width, int height, thrust::device_ptr<int const> const weights) {
    auto x = blockIdx.x * blockDim.x + threadIdx.x + 2;
    auto y = blockIdx.y * blockDim.y + threadIdx.y + 2;

    auto index = y * width + x;

    if (x >= width - 2 || y >= height - 2) {
        return;
    }

    auto sum =
        weights[0 * 5 + 2] +
        weights[1 * 5 + 2] +
        weights[2 * 5 + 0] +
        weights[2 * 5 + 1] +
        weights[2 * 5 + 2] +
        weights[2 * 5 + 3] +
        weights[2 * 5 + 4] +
        weights[3 * 5 + 2] +
        weights[4 * 5 + 2]
    ;

    output[index] = 1.0f / sum * (
        weights[0 * 5 + 2] * input[index - width - width] +
        weights[1 * 5 + 2] * input[index - width] +
        weights[2 * 5 + 0] * input[index - 1 - 1] +
        weights[2 * 5 + 1] * input[index - 1] +
        weights[2 * 5 + 2] * input[index] +
        weights[2 * 5 + 3] * input[index + 1] +
        weights[2 * 5 + 4] * input[index + 1 + 1] +
        weights[3 * 5 + 2] * input[index + width] +
        weights[4 * 5 + 2] * input[index + width + width]
    );
}
