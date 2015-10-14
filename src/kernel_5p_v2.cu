#include <thrust/device_ptr.h>

__global__ void fivePoint2(thrust::device_ptr<float const> const input, thrust::device_ptr<float> const output, int width, int height) {
    auto x = blockIdx.x * blockDim.x + threadIdx.x + 1;
    auto y = blockIdx.y * blockDim.y + threadIdx.y + 1;

    auto index = y * width + x;

    output[index] = 1.0f / 5 * (input[index - width] + input[index - 1] + input[index] + input[index + 1] + input[index + width]);
}
